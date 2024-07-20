import cv2
import pytesseract

img = cv2.imread("hello.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("hello", img)
cv2.waitKey(0)


pytesseract.pytesseract.tesseract_cmd = r"C:\Users\KEVISH\AppData\Local\Programs\Tesseract-OCR\\tesseract.exe"

h, w, _ = img.shape

boxes_str: str = pytesseract.image_to_boxes(img)

for box in boxes_str.splitlines():
    box = box.split(" ")
    x1, y1, x2, y2 = map(int, box[1:5])
    y1, y2 = h-y1, h-y2
    print(x1, y1, x2, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
    cv2.putText(img, box[0], (x,))

cv2.imshow("hello", img)
cv2.waitKey(0)

