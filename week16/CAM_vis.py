import cv2
import matplotlib.pyplot as plt
import numpy as np

class CAM_localization:

    def _norm(self, img, max_val=255.0):
        if max_val == 255.0:
            return np.array(
                np.multiply(np.subtract(img, np.min(img)), max_val / (np.max(img) - np.min(img))))  # .astype(np.uint8)
        else:
            return np.array(np.multiply(np.subtract(img, np.min(img)), max_val / (np.max(img) - np.min(img))))

    def get_localization_map(self, image, feature_map, show_result=True):
        imge = self._norm(image).astype(np.uint8)
        vis = cv2.resize(feature_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        # vis = vis - vis.min()
        if show_result:
            plt.imshow(imge)
            plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
            plt.show()
            plt.cla()

        vis2 = self._norm(vis).astype(np.uint8)
        vis2 = cv2.GaussianBlur(vis2, (5, 5), 0)
        ret, imgf = cv2.threshold(vis2, 0, np.max(vis2) / 2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x, y, w, h = cv2.boundingRect(imgf.copy())
        cv2.rectangle(imge, (x, y), (x + w, y + h), (0, 255, 0),
                      max(int(max(image.shape[1], image.shape[0]) * 0.006), 2))
        if show_result:
            plt.imshow(imge)
            plt.show()
            plt.cla()

        return