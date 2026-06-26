from scripts.normalize.schema import UnifiedExample
from scripts.normalize.task_type import infer_task_type


def test_task_type_infer():
    assert infer_task_type("Point to the cup left of the laptop") == "spatial"
    assert infer_task_type("Point to where a small item could be placed") == "free_space"
    assert infer_task_type("Point to something used to stir soup") == "affordance"


def test_schema_minimal():
    ex = UnifiedExample(
        uid="a",
        dataset="pixmo_points",
        split="train",
        image_path="x.jpg",
        image_width=100,
        image_height=100,
        query="Point to the mug",
        task_type="pointing",
        points=[{"x": 0.5, "y": 0.5}],
    )
    assert ex.uid == "a"
