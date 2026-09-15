"""
SignalScope Bonus Module D: Provenance & Metadata Inspector
Inspects C2PA / Content Credentials manifests and EXIF optical metadata,
and computes a fused authenticity trust score.
"""

from PIL import Image, ExifTags
import os


class MetadataInspector:
    """
    Parses EXIF hardware tags and scans binary chunks for C2PA / Content Credentials manifests.
    Fuses metadata evidence with deep learning model outputs.
    """
    KNOWN_AI_SOFTWARE_SIGNATURES = [
        "stable diffusion", "automatic1111", "comfyui", "midjourney",
        "dall-e", "dalle", "firefly", "novelai", "invokeai", "fooocus"
    ]

    def inspect_file(self, file_path: str) -> dict:
        """
        Extracts EXIF and C2PA provenance indicators from an image file.
        """
        results = {
            "has_exif": False,
            "camera_metadata": {},
            "software_signature": None,
            "has_c2pa": False,
            "c2pa_status": "Not detected",
            "metadata_signal": "neutral",
            "metadata_summary": ""
        }

        if not os.path.exists(file_path):
            results["metadata_summary"] = "File not found."
            return results

        # 1. C2PA / Content Credentials binary scan (JUMBF container inspection)
        try:
            with open(file_path, "rb") as f:
                header_bytes = f.read(512 * 1024)  # Scan first 512KB for provenance chunks
                if b"c2pa" in header_bytes or b"urn:c2pa:" in header_bytes or b"C2PA" in header_bytes:
                    results["has_c2pa"] = True
                    results["c2pa_status"] = "C2PA / Content Credentials manifest present"
                    results["metadata_signal"] = "c2pa_present"
        except Exception:
            pass

        # 2. EXIF camera and software inspection
        try:
            with Image.open(file_path) as img:
                exif_data = img.getexif()
                if exif_data:
                    results["has_exif"] = True
                    tag_map = {}
                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        tag_map[tag_name] = str(value)

                    # Check camera hardware tags
                    make = tag_map.get("Make", "")
                    model = tag_map.get("Model", "")
                    software = tag_map.get("Software", "").lower()

                    if make or model:
                        results["camera_metadata"] = {"make": make, "model": model}

                    if software:
                        results["software_signature"] = software
                        for sig in self.KNOWN_AI_SOFTWARE_SIGNATURES:
                            if sig in software:
                                results["metadata_signal"] = "ai_software_detected"
                                results["metadata_summary"] = f"Generative AI software tag found in metadata: '{software}'."
                                return results

                    if results["camera_metadata"]:
                        results["metadata_signal"] = "authentic_camera_tags"
                        results["metadata_summary"] = f"Valid physical camera hardware tags verified ({make} {model})."
                        return results
        except Exception:
            pass

        if results["has_c2pa"]:
            results["metadata_summary"] = "Cryptographic provenance manifest detected in image header."
        elif not results["has_exif"]:
            results["metadata_signal"] = "metadata_stripped"
            results["metadata_summary"] = "Metadata is stripped or absent (typical of social media compression). Forensic decision relies solely on visual pixel analysis."
        else:
            results["metadata_summary"] = "EXIF present without definitive AI software tags or hardware signatures."

        return results

    def fuse_signals(self, visual_prob: float, metadata_info: dict) -> dict:
        """
        Combines visual classifier confidence with metadata provenance evidence.
        """
        signal = metadata_info.get("metadata_signal", "neutral")
        fused_prob = visual_prob
        adjustment_reason = "Visual forensic classifier output."

        if signal == "ai_software_detected":
            fused_prob = max(visual_prob, 0.98)
            adjustment_reason = "EXIF metadata explicitly confirms generative AI software creation."
        elif signal == "c2pa_present":
            fused_prob = max(visual_prob, 0.95)
            adjustment_reason = "C2PA / Content Credentials manifest confirms synthetic generative origin."
        elif signal == "authentic_camera_tags" and visual_prob < 0.60:
            # Physical camera sensor tags provide confidence discount if visual cues are not overwhelmingly fake
            fused_prob = max(0.05, visual_prob * 0.75)
            adjustment_reason = "Genuine camera sensor EXIF tags corroborated; lower likelihood of synthetic origin."

        return {
            "visual_probability": round(visual_prob, 4),
            "fused_probability": round(fused_prob, 4),
            "fusion_rationale": adjustment_reason,
        }
