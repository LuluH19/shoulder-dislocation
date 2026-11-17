import numpy as np

from gestures.config import CLASSES
from gestures.data import infer_class, infer_group, load_recordings, read_csv

ROWS = np.arange(75 * 6, dtype=np.float32).reshape(75, 6)


def write(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(",".join(f"{v}" for v in row) for row in ROWS)
    path.write_text((header + "\n" if header else "") + body + "\n")


def test_all_header_styles_give_the_same_channel_order(tmp_path):
    styles = {
        "a.csv": "aX,aY,aZ,gX,gY,gZ",
        "b.csv": "acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z",
        "c.csv": "X,Y,Z,GX,GY,GZ",
        "d.csv": "",
    }
    for name, header in styles.items():
        write(tmp_path / name, header)
        assert np.array_equal(read_csv(tmp_path / name), ROWS), name


def test_reordered_columns_are_put_back_in_order(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text("gZ,aX,aY,aZ,gX,gY\n" + "\n".join(",".join(str(v) for v in np.roll(r, 1)) for r in ROWS))
    assert np.allclose(read_csv(path), ROWS)


def test_class_and_group_come_from_folder_and_file_names(tmp_path):
    assert infer_class(tmp_path / "group1" / "vert" / "1_vert_3.csv") == "vertical"
    assert infer_class(tmp_path / "group5" / "hor" / "5_hor_1.csv") == "horizontal"
    assert infer_class(tmp_path / "x" / "3_Circle_1.csv") == "circle"
    assert infer_group(tmp_path / "group3" / "circle" / "3_Circle_1.csv") == "group3"
    assert infer_group(tmp_path / "ailleurs" / "2_Hor_7.csv") == "group2"


def test_load_recordings_skips_duplicates_and_unknown_files(tmp_path):
    write(tmp_path / "group1" / "circle" / "1_circle_1.csv", "aX,aY,aZ,gX,gY,gZ")
    write(tmp_path / "group1" / "hor" / "1_hor_1.csv", "aX,aY,aZ,gX,gY,gZ")
    write(tmp_path / "copie" / "circle" / "1_circle_1.csv", "aX,aY,aZ,gX,gY,gZ")  # doublon
    write(tmp_path / "group1" / "autre" / "1_autre_1.csv", "aX,aY,aZ,gX,gY,gZ")   # geste inconnu
    rec = load_recordings(tmp_path, verbose=False)
    assert rec.X.shape == (2, 75, 6)
    assert sorted(CLASSES[k] for k in rec.y) == ["circle", "horizontal"]
    assert set(rec.groups) == {"group1"}
