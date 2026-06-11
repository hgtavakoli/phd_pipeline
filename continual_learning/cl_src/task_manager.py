# cl_src/task_manager.py
import re

class TaskManager:
    def __init__(self, task_sequence_list: list, all_classes: list):
        """
        Initializes the TaskManager.
        The task_sequence is a list of strings, where each string contains
        the classes for one task, separated by commas or semicolons.
        """
        self.tasks = []
        for task_string in task_sequence_list:
            # Use regex to split by either comma or semicolon, and remove empty strings
            classes = [c.strip() for c in re.split('[,;]', task_string) if c.strip()]
            self.tasks.append(classes)

        self.all_classes = all_classes
        self.task_id_counter = 0
        self.seen_classes = []

    def get_task_info(self):
        """
        A generator that yields information for each task sequentially.
        Crucially, it yields the class_offset needed for re-labeling.
        """
        self.seen_classes = []
        while self.task_id_counter < len(self.tasks):
            # Calculate the offset BEFORE adding the new classes
            class_offset = len(self.seen_classes)

            current_task_classes = self.tasks[self.task_id_counter]
            self.seen_classes.extend(c for c in current_task_classes if c not in self.seen_classes)
            seen_classes_so_far = set(self.seen_classes)
            ordered_seen_classes = [c for c in self.all_classes if c in seen_classes_so_far]

            yield self.task_id_counter + 1, current_task_classes, ordered_seen_classes, class_offset

            self.task_id_counter += 1