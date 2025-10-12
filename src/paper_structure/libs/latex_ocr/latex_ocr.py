"""Main LaTeX OCR class."""

import re
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

from .config import LaTeXOCRConfig
from .models import EncoderDecoder
from .onnx_session import OrtInferSession
from .preprocess import PreProcess
from .tokenizer import TokenizerCls
from .utils import ModelDownloader


class LaTeXOCR:
    """LaTeX OCR for formula recognition with GPU/TensorRT support.
    
    Automatically downloads models on first use from GitHub releases.
    Supports hardware acceleration: TensorRT > CUDA > CPU
    """
    
    def __init__(
        self,
        config: LaTeXOCRConfig = None,
        image_resizer_path: Union[str, Path] = None,
        encoder_path: Union[str, Path] = None,
        decoder_path: Union[str, Path] = None,
        tokenizer_json: Union[str, Path] = None,
        use_gpu: bool = False,
        use_tensorrt: bool = False,
    ):
        """Initialize LaTeX OCR.
        
        Args:
            config: Configuration object (uses defaults if None)
            image_resizer_path: Path to image resizer ONNX model
            encoder_path: Path to encoder ONNX model
            decoder_path: Path to decoder ONNX model
            tokenizer_json: Path to tokenizer JSON file
            use_gpu: Enable CUDA GPU acceleration
            use_tensorrt: Enable TensorRT acceleration (requires TensorRT)
        """
        if config is None:
            config = LaTeXOCRConfig()
        
        self.config = config
        self.use_gpu = use_gpu
        self.use_tensorrt = use_tensorrt
        
        # Download and get model paths
        self.image_resizer_path = image_resizer_path
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.tokenizer_json = tokenizer_json
        self._get_model_paths()
        
        # Initialize components
        self.max_dims = [config.max_width, config.max_height]
        self.min_dims = [config.min_width, config.min_height]
        self.temperature = config.temperature
        
        self.pre_pro = PreProcess(max_dims=self.max_dims, min_dims=self.min_dims)
        
        self.image_resizer = OrtInferSession(
            self.image_resizer_path,
            use_gpu=use_gpu,
            use_tensorrt=use_tensorrt,
        )
        
        self.encoder_decoder = EncoderDecoder(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            bos_token=config.bos_token,
            eos_token=config.eos_token,
            max_seq_len=config.max_seq_len,
            use_gpu=use_gpu,
            use_tensorrt=use_tensorrt,
        )
        
        self.tokenizer = TokenizerCls(self.tokenizer_json)
    
    def _get_model_paths(self):
        """Download models if needed and get their paths."""
        downloader = ModelDownloader()
        
        # Model file names
        decoder_name = "decoder.onnx"
        encoder_name = "encoder.onnx"
        resizer_name = "image_resizer.onnx"
        tokenizer_name = "tokenizer.json"
        
        # Download if paths not provided
        if self.image_resizer_path is None:
            self.image_resizer_path = downloader.download(resizer_name)
        
        if self.encoder_path is None:
            self.encoder_path = downloader.download(encoder_name)
        
        if self.decoder_path is None:
            self.decoder_path = downloader.download(decoder_name)
        
        if self.tokenizer_json is None:
            self.tokenizer_json = downloader.download(tokenizer_name)
    
    def __call__(self, img: Union[np.ndarray, Image.Image, str, Path]) -> Tuple[str, float]:
        """Recognize LaTeX formula from image.
        
        Args:
            img: Input image (numpy array, PIL Image, or path to image file)
            
        Returns:
            Tuple of (latex_string, elapsed_time)
        """
        s = time.perf_counter()
        
        # Load image
        if isinstance(img, (str, Path)):
            img = np.array(Image.open(img))
        elif isinstance(img, Image.Image):
            img = np.array(img)
        
        # Convert to BGR if needed
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Resize image
        resized_img = self._loop_image_resizer(img)
        
        # Encode and decode
        dec = self.encoder_decoder(resized_img, temperature=self.temperature)
        
        # Convert tokens to LaTeX string
        decode = self.tokenizer.token2str(dec)
        pred = self._post_process(decode[0])
        
        elapse = time.perf_counter() - s
        return pred, elapse
    
    def _loop_image_resizer(self, img: np.ndarray) -> np.ndarray:
        """Iteratively resize image to optimal width.
        
        Args:
            img: Input image array
            
        Returns:
            Resized image array
        """
        pillow_img = Image.fromarray(img)
        pad_img = self.pre_pro.pad(pillow_img)
        input_image = self.pre_pro.minmax_size(pad_img).convert("RGB")
        r, w, h = 1, input_image.size[0], input_image.size[1]
        
        for _ in range(10):
            h = int(h * r)
            final_img, pad_img = self._pre_process(input_image, r, w, h)
            
            resizer_res = self.image_resizer([final_img.astype(np.float32)])[0]
            
            argmax_idx = int(np.argmax(resizer_res, axis=-1))
            w = (argmax_idx + 1) * 32
            
            if w == pad_img.size[0]:
                break
            
            r = w / pad_img.size[0]
        
        return final_img
    
    def _pre_process(
        self, 
        input_image: Image.Image, 
        r: float, 
        w: int, 
        h: int
    ) -> Tuple[np.ndarray, Image.Image]:
        """Preprocess image for model input.
        
        Args:
            input_image: Input PIL Image
            r: Resize ratio
            w: Target width
            h: Target height
            
        Returns:
            Tuple of (processed_array, padded_image)
        """
        if r > 1:
            resize_func = Image.Resampling.BILINEAR
        else:
            resize_func = Image.Resampling.LANCZOS
        
        resize_img = input_image.resize((w, h), resize_func)
        pad_img = self.pre_pro.pad(self.pre_pro.minmax_size(resize_img))
        cvt_img = np.array(pad_img.convert("RGB"))
        
        gray_img = self.pre_pro.to_gray(cvt_img)
        normal_img = self.pre_pro.normalize(gray_img)
        final_img = self.pre_pro.transpose_and_four_dim(normal_img)
        return final_img, pad_img
    
    @staticmethod
    def _post_process(s: str) -> str:
        """Remove unnecessary whitespace from LaTeX code.
        
        Args:
            s: Input LaTeX string
            
        Returns:
            Processed LaTeX string
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s
