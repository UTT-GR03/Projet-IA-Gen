import os
from io import BytesIO
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class RealESRGAN:

    def inference(
        self,
        input_image: BytesIO,
        model_name: str='RealESRGAN_x4plus',
        denoise_strength: int=.5,
        model_path: str=None,
        outscale: float=4,
        tile: int=0,
        tile_pad: int=10,
        pre_pad: int=0,
        face_enhance: bool=False,
        fp32: bool=False,
        gpu_id=None,
    ):
        """
        Function used for upscaling an input image with a Real-ESRGAN model.

        Parameters
        ----------
        input_image: _io.BytesIO
            Image to upscale
        model_name: string
            default='RealESRGAN_x4plus'
            Name of the model to use.
            Model names:
                RealESRGAN_x4plus
                RealESRNet_x4plus
                RealESRGAN_x4plus_anime_6B
                RealESRGAN_x2plus
                realesr-animevideov3
                realesr-general-x4v3
        denoise_strength: float
            default=.5
            Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability.
            Only used for the realesr-general-x4v3 model
        model_path: string
            default=None
            Model path. Usually, you do not need to specify it
        outscale: float
            default=4,
            The final upsampling scale of the image
        tile: int
            default=0
            Tile size, 0 for no tile during testing
        tile_pad: int
            default=10
            Tile padding
        pre_pad: int
            default=0
            Pre padding size at each border
        face_enhance: bool
            default=False
            Use GFPGAN to enhance face
        gpu_id: int
            default=None
            gpu device to use (default=None) can be 0,1,2 for multi-gpu')

        Returns
        -------
        upscaled_image
        """

        # determine models according to model names
        model_name = model_name.split('.')[0]
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # determine model paths
        if model_path is not None:
            model_path = model_path
        else:
            model_path = os.path.join('realesrgan_weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'realesrgan_weights'), progress=True, file_name=None)

        # use dni to control the denoise strength
        dni_weight = None
        if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]

        # restorer
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=gpu_id)

        if face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)

        img = input_image

        try:
            if face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        
        return output