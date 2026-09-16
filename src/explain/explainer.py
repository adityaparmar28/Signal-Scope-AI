import numpy as np

def analyze_heatmap(heatmap: np.ndarray) -> dict:
    threshold = 0.5
    h, w = heatmap.shape
    hotspots = heatmap > threshold
    
    num_hotspots = int(np.sum(hotspots))
    max_intensity = float(np.max(heatmap))
    mean_intensity = float(np.mean(heatmap))
    
    quadrants = {
        'top-left': heatmap[:h//2, :w//2],
        'top-right': heatmap[:h//2, w//2:],
        'bottom-left': heatmap[h//2:, :w//2],
        'bottom-right': heatmap[h//2:, w//2:],
        'center': heatmap[h//4:3*h//4, w//4:3*w//4]
    }
    
    regions = []
    total_area = h * w
    for pos, area in quadrants.items():
        area_hotspots = area > threshold
        area_fraction = float(np.sum(area_hotspots) / total_area)
        if area_fraction > 0.01:
            regions.append({
                'position': pos,
                'intensity': float(np.mean(area[area_hotspots])) if np.any(area_hotspots) else 0.0,
                'area_fraction': area_fraction
            })
            
    is_localized = num_hotspots > 0 and (num_hotspots / total_area) < 0.2
    
    return {
        'num_hotspots': num_hotspots,
        'regions': regions,
        'is_localized': is_localized,
        'max_intensity': max_intensity,
        'mean_intensity': mean_intensity
    }

def generate_explanation(confidence: float, label: str, heatmap_analysis: dict) -> str:
    explanation = ""
    if label.lower() == "ai-generated" or label.lower() == "fake":
        if heatmap_analysis['is_localized'] and heatmap_analysis['regions']:
            pos = heatmap_analysis['regions'][0]['position']
            explanation = f"The model identified specific anomalous regions in the {pos} of the image, suggesting potential artifacts in texture or geometry. "
        else:
            explanation = "The model detected widespread subtle inconsistencies across the image, consistent with synthetic generation patterns. "
            
        if confidence > 0.8:
            explanation += "Analysis indicates strong synthetic artifact signatures."
        else:
            explanation += "The evidence is inconclusive. The image shows some characteristics that could indicate synthetic origin, but confidence is low."
    else:
        explanation = "The image appears consistent with natural photographic characteristics. No significant synthetic artifacts were detected."
        
    explanation += "\nNote: This is a probabilistic assessment, not a definitive determination."
    return explanation

def generate_full_report(prediction: dict, heatmap: np.ndarray) -> dict:
    heatmap_analysis = analyze_heatmap(heatmap)
    explanation = generate_explanation(prediction.get('confidence', 0.0), prediction.get('label', 'real'), heatmap_analysis)
    
    return {
        'verdict': prediction.get('verdict', 'unknown'),
        'confidence': prediction.get('confidence', 0.0),
        'explanation': explanation,
        'heatmap_analysis': heatmap_analysis,
        'disclaimer': "Note: This is a probabilistic assessment, not a definitive determination."
    }
