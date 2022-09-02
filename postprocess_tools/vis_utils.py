import cv2

colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


def get_text_pos_fn2(pmin, pmax, bbox, label):
    p1 = (pmax[0], pmax[1])
    return p1[0] - 5, p1[1]


def draw_bbox(img, bbox, shape=None, label=None, color=(255, 0, 0), thickness=2, is_relative_bbox=False, xy_order=True,
              xywh=False):
    if is_relative_bbox:
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    else:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
    if xywh:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    if xy_order:
        p1 = p1[::-1]
        p2 = p2[::-1]
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0] - 5, p1[1])
    if label is not None:
        cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
    return img


def color_fn(label):
    color_nr = len(colors_tableau)
    return colors_tableau[label % color_nr]
