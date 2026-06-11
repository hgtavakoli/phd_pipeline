import time
import asyncio
import json
import io

import omni.kit
import omni.usd
import omni.replicator.core as rep

from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch

class WorkerWriter(Writer):
    def __init__(
        self,
        output_dir,
        rgb: bool = True,
        bounding_box_2d_tight: bool = False,
        image_output_format="png",
    ):
        self._output_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0
        self._image_output_format = image_output_format

        self.annotators = []

        # RGB
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))

        # Bounding Box 2D
        if bounding_box_2d_tight:
            self.annotators.append(AnnotatorRegistry.get_annotator("bounding_box_2d_tight",
                                                                init_params={"semanticTypes": ["class"]}))

    def check_bbox_area(self, bbox_data, size_limit):
        # Note: This logic seems to assume a single bounding box, may need adjustment for multiple objects
        length = abs(bbox_data['x_min'][0] - bbox_data['x_max'][0])
        width = abs(bbox_data['y_min'][0] - bbox_data['y_max'][0])

        area = length * width
        if area > size_limit:
            return True
        else:
            return False

    def write(self, data):
        # This writer only activates if both rgb and bounding box data are present
        if "rgb" in data and "bounding_box_2d_tight" in data:
            bbox_data_all = data["bounding_box_2d_tight"]["data"]
            id_to_labels = data["bounding_box_2d_tight"]["info"]["idToLabels"]

            # Loop through all detected objects to see if any is a 'worker'
            for i, bbox_record in enumerate(bbox_data_all):
                semantic_id = str(bbox_record['semanticId'])
                
                if semantic_id in id_to_labels and 'GlueGun' in id_to_labels[semantic_id]:
                    
                    # Check if the worker's bounding box is large enough
                    if self.check_bbox_area(bbox_record, 0.5):
                        width = int(bbox_record["x_max"] - bbox_record["x_min"])
                        height = int(bbox_record["y_max"] - bbox_record["y_min"])
                        
                        # Check for invalid bounding box dimensions
                        if width > 0 and height > 0:
                            filepath = f"rgb_{self._frame_id}.{self._image_output_format}"
                            self._backend.write_image(filepath, data["rgb"])

                            bbox_filepath = f"bbox_{self._frame_id}.json"

                            # Save this specific bounding box in COCO-like format
                            coco_bbox_data = {
                                "x": int(bbox_record["x_min"]),
                                "y": int(bbox_record["y_min"]),
                                "width": width,
                                "height": height
                            }

                            buf = io.BytesIO()
                            buf.write(json.dumps(coco_bbox_data).encode())
                            self._backend.write_blob(bbox_filepath, buf.getvalue())

                            # We found and saved a worker, so we can stop processing this frame
                            self._frame_id += 1
                            return

# =======================================================================
# V V V V V V V V V V   ADD THIS LINE AT THE END   V V V V V V V V V V V V
# =======================================================================
rep.WriterRegistry.register(WorkerWriter)
# =======================================================================