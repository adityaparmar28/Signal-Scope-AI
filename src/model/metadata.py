import piexif
from PIL import Image

def analyze_metadata(image_path: str) -> dict:
    """
    Extracts EXIF and provenance metadata from an image.
    Looks for signatures of AI generation (e.g., software tags, missing standard camera data).
    """
    result = {
        'has_exif': False,
        'camera_make': None,
        'camera_model': None,
        'software': None,
        'ai_signature_found': False,
        'suspicious': False,
        'raw_metadata': {}
    }
    
    try:
        img = Image.open(image_path)
        if 'exif' not in img.info:
            result['suspicious'] = True  # Real photos almost always have EXIF, generators often strip it or don't add it
            return result
            
        result['has_exif'] = True
        exif_dict = piexif.load(img.info['exif'])
        
        # 0th IFD (Image File Directory) contains main image info
        if "0th" in exif_dict:
            # 271 = Make, 272 = Model, 305 = Software
            if 271 in exif_dict["0th"]:
                result['camera_make'] = exif_dict["0th"][271].decode('utf-8', errors='ignore').strip().strip('\x00')
            if 272 in exif_dict["0th"]:
                result['camera_model'] = exif_dict["0th"][272].decode('utf-8', errors='ignore').strip().strip('\x00')
            if 305 in exif_dict["0th"]:
                result['software'] = exif_dict["0th"][305].decode('utf-8', errors='ignore').strip().strip('\x00')
                
        # Check for known AI signatures in the software tag
        ai_keywords = ['midjourney', 'dall-e', 'stable diffusion', 'comfyui', 'automatic1111', 'ai', 'generative']
        if result['software']:
            software_lower = result['software'].lower()
            if any(keyword in software_lower for keyword in ai_keywords):
                result['ai_signature_found'] = True
                
        # Lack of camera make/model but presence of EXIF is highly suspicious
        if not result['camera_make'] and not result['camera_model']:
            result['suspicious'] = True
            
        # Add basic tags to raw metadata for UI display
        if result['camera_make']: result['raw_metadata']['Make'] = result['camera_make']
        if result['camera_model']: result['raw_metadata']['Model'] = result['camera_model']
        if result['software']: result['raw_metadata']['Software'] = result['software']
        
    except Exception as e:
        # If there's any error parsing, treat it gracefully
        pass
        
    return result
