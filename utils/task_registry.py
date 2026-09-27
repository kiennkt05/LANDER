"""Stable, half-open class ranges in the model's remapped label space."""


class TaskRegistry:
    def __init__(self, ranges=()):
        self.ranges = []
        for row in ranges:
            self.append(*row)

    def append(self, task_id, start, end):
        if task_id != len(self.ranges) or start != (self.ranges[-1][2] if self.ranges else 0) or end <= start:
            raise ValueError("task ranges must be consecutive, nonempty and disjoint")
        self.ranges.append((int(task_id), int(start), int(end)))

    def task_for(self, class_id):
        for task, start, end in self.ranges:
            if start <= class_id < end:
                return task
        raise KeyError(class_id)
