import cv2


def tag_faces(image, classified_faces, rects):
    for i, (label, _) in enumerate(classified_faces):
        x1, y1, x2, y2 = rects[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
