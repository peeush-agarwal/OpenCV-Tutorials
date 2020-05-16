import cv2

def apply_thresholding(cap):
    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        greyed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur of 5,5 kernel and 0=SigmaX
        greyed = cv2.GaussianBlur(greyed, (5,5), 0)
        # Simple thresholding on frame. Pixel values greater than 127 would be assigned 255.
        _, simple_th = cv2.threshold(greyed, 127, 255, cv2.THRESH_BINARY)
        # Simple thresholding on frame. Pixel values less than 127 would be assigned 255.
        _, simple_th_inv = cv2.threshold(greyed, 127, 255, cv2.THRESH_BINARY_INV)

        # Adaptive thresholding: Pixels value is adapted by taking mean of neighborhood pixels
        ada_th_mean = cv2.adaptiveThreshold(greyed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # Adaptive thresholding: Pixels value is adapted by taking Gaussian mean of neighborhood pixels
        ada_th_gaussian = cv2.adaptiveThreshold(greyed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # OTSU's thresholding
        retVal, otsu_th = cv2.threshold(greyed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print(f'OTSU TH: {retVal}')

        cv2.imshow('Original', frame)
        cv2.imshow('Simple TH Binary', simple_th)
        cv2.imshow('Simple TH BinInv', simple_th_inv)
        cv2.imshow('Adaptive TH Mean', ada_th_mean)
        cv2.imshow('Adaptive TH Gaussian', ada_th_gaussian)
        cv2.imshow('OTSU TH', otsu_th)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break