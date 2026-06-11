import omni.replicator.core as rep
import numpy as np
import os

class YOLOWriter(rep.Writer):
    VERSION = "1.0.0"

    def __init__(self, output_dir: str, class_mapping: dict):
        self.output_dir = output_dir
        self.class_mapping = {name.lower(): id for name, id in class_mapping.items()}
        self.annotators = ["rgb", "bounding_box_2d_tight"]
        print(f"YOLOWriter Initialized. Output directory: {self.output_dir}")
        print(f"Class Mapping: {self.class_mapping}")

    def write(self, data: dict):
        """
        This is the core method called by the orchestrator for each rendered frame.
        """
        bbox_data = data["bounding_box_2d_tight"]
        bbox_info = bbox_data["info"]
        
        render_product_name = bbox_info["render_product_name"]
        frame_num = bbox_info["frame"]
        
        rp_output_dir = os.path.join(self.output_dir, render_product_name)
        os.makedirs(rp_output_dir, exist_ok=True)

        # --- Save the Image ---
        img_data = data["rgb"]
        img_height, img_width, _ = img_data.shape
        img_filename = f"frame_{frame_num:04d}.png"
        img_filepath = os.path.join(rp_output_dir, img_filename)
        rep.utils.save_image(img_data, img_filepath)

        # --- Process Bounding Boxes and Save YOLO Label ---
        yolo_lines = []
        if bbox_data["data"].size > 0:
            semantic_labels = bbox_info["semanticLabels"]
            
            for bbox in bbox_data["data"]:
                semantic_id = bbox['semanticId']
                class_name = semantic_labels.get(str(semantic_id), {}).get('class', '').lower()

                if class_name in self.class_mapping:
                    class_id = self.class_mapping[class_name]
                    
                    x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
                    
                    box_width = float(x_max - x_min)
                    box_height = float(y_max - y_min)
                    x_center = float(x_min + (box_width / 2))
                    y_center = float(y_min + (box_height / 2))

                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = box_width / img_width
                    height_norm = box_height / img_height
                    
                    yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                    yolo_lines.append(yolo_line)

        label_filename = f"frame_{frame_num:04d}.txt"
        label_filepath = os.path.join(rp_output_dir, label_filename)
        with open(label_filepath, 'w') as f:
            f.write("\n".join(yolo_lines))
    
    # ====================================================================
    # === FIX IS HERE: Define the function with no arguments (besides self) ===
    # ====================================================================
    def write_metadata(self):
        """
        Override the default metadata writer to do nothing.
        This signature matches the internal call `self.write_metadata()`.
        """
        pass