import cv2
import mediapipe as mp
import itertools
import numpy as np
from scipy.interpolate import splev, splprep

class MakeupApplication:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize mediapipe solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=min_detection_confidence,
                                                    min_tracking_confidence=min_tracking_confidence)

        # Precompute index lists for facial landmarks
        self.LEFT_EYE_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYE)))
        self.RIGHT_EYE_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYE)))
        self.LIPS_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LIPS)))
        self.LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
        self.RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))

    def get_upper_side_coordinates(self, eye_landmarks):
        sorted_landmarks = sorted(eye_landmarks, key=lambda coord: coord.y)
        half_length = len(sorted_landmarks) // 2
        return sorted_landmarks[:half_length]

    def get_lower_side_coordinates(self, eye_landmarks):
        sorted_landmarks = sorted(eye_landmarks, key=lambda coord: coord.y)
        half_length = len(sorted_landmarks) // 2
        return sorted_landmarks[half_length:]
    

    def apply_lipstick(self, image, landmarks, indexes, color, blur_kernel_size=(7, 7), blur_sigma=10, color_intensity=0.4):
        points = np.array([(int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0])) for idx in indexes])

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [cv2.convexHull(points)], 255)

        boundary_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        colored_image = np.zeros_like(image)
        colored_image[:] = color

        lipstick_image = cv2.bitwise_and(colored_image, colored_image, mask=boundary_mask)
        lips_colored = cv2.addWeighted(image, 1, lipstick_image, color_intensity, 0)

        blurred = cv2.GaussianBlur(lips_colored, blur_kernel_size, blur_sigma)

        gradient_mask = cv2.GaussianBlur(boundary_mask, (15, 15), 0)
        gradient_mask = gradient_mask / 255.0
        lips_with_gradient = (blurred * gradient_mask[..., np.newaxis] + image * (1 - gradient_mask[..., np.newaxis])).astype(np.uint8)

        final_image = np.where(boundary_mask[..., np.newaxis] == 0, image, lips_with_gradient)
        return final_image


    def draw_eyeliner(self, image, upper_eye_coordinates, color=(14, 14, 18), thickness=2, fade_factor=0.6):
        result_image = image.copy()
        
        # Convert eye coordinates to pixel values
        eyeliner_points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in upper_eye_coordinates]
        eyeliner_points.sort(key=lambda x: x[0])
        eyeliner_points = np.array(eyeliner_points, dtype=np.int32)

        # Create eyeliner path using a smooth spline curve for more natural shape
        if len(eyeliner_points) >= 4:
            # Fit a spline through the eyeliner points for a smooth contour
            spline_curve = cv2.polylines(np.zeros_like(image), [eyeliner_points], isClosed=False, color=(255, 255, 255), thickness=thickness)

        for i in range(len(eyeliner_points) - 1):
            start_point = tuple(eyeliner_points[i])
            end_point = tuple(eyeliner_points[i + 1])

            # Set thickness to be dynamic for realism (thicker near center)
            relative_pos = i / (len(eyeliner_points) - 1)
            dynamic_thickness = int(thickness * (1 - fade_factor * abs(relative_pos - 0.5)))

            # Draw line segments with dynamic thickness along curve
            cv2.line(result_image, start_point, end_point, color, dynamic_thickness)

        # Optional: Apply slight blur to soften edges for a more blended effect
        blurred_result = cv2.GaussianBlur(result_image, (3, 3), 1)

        # Blend original image with the blurred eyeliner for natural integration
        final_result = cv2.addWeighted(result_image, 0.9, blurred_result, 0.1, 0)

        return final_result


     

    def apply_eyeshadow(self, frame, left_eye_indices, right_eye_indices, color=(130, 50, 200), alpha=0.17, blur_radius=25):
        # Convert the frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame) 

        if result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Create a base mask to hold the eyeshadow effect
            mask = np.zeros_like(frame, dtype=np.uint8)

            for face_landmarks in result.multi_face_landmarks:
                # Function to create and apply eyeshadow within the specific eye region
                def create_eyeshadow(eye_indices):
                    # Extract eye region coordinates
                    eye_points = np.array([
                        [
                            int(face_landmarks.landmark[idx].x * w),
                            int(face_landmarks.landmark[idx].y * h)
                        ]
                        for idx in eye_indices
                    ], np.int32)

                    # Create an inner region mask (within the provided eye coordinates)
                    eye_mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillPoly(eye_mask, [eye_points], color)

                    # Overlay only the inner mask onto the main mask to restrict effect to the inside of coordinates
                    mask[:, :, 0] = np.where(eye_mask[:, :, 0] > 0, color[0], mask[:, :, 0])
                    mask[:, :, 1] = np.where(eye_mask[:, :, 1] > 0, color[1], mask[:, :, 1])
                    mask[:, :, 2] = np.where(eye_mask[:, :, 2] > 0, color[2], mask[:, :, 2])

                # Apply the eyeshadow to both eyes
                create_eyeshadow(left_eye_indices)
                create_eyeshadow(right_eye_indices)

            # Apply Gaussian blur to soften the edges of the mask
            blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

            # Blend the blurred mask with the original image
            frame = cv2.addWeighted(blurred_mask, alpha, frame, 1 - alpha, 0)

        return frame



    def apply_blush(self, frame, left_cheek_indices, right_cheek_indices, color=(130, 119, 255), alpha=0.12, blur_radius=25):
        # Convert the frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Create a base mask to hold the blush effect
            mask = np.zeros_like(frame, dtype=np.uint8)

            for face_landmarks in result.multi_face_landmarks:
                def create_gradient_blush(cheek_indices):
                    # Get cheek coordinates and determine center
                    cheek_points = np.array([
                        [
                            int(face_landmarks.landmark[idx].x * w),
                            int(face_landmarks.landmark[idx].y * h)
                        ]
                        for idx in cheek_indices
                    ], np.int32)
                    
                    # Calculate center of cheek
                    cheek_center = np.mean(cheek_points, axis=0).astype(int)
                    max_distance = np.max(np.linalg.norm(cheek_points - cheek_center, axis=1))

                    # Create a local cheek mask
                    cheek_mask = np.zeros((h, w), dtype=np.float32)
                    cv2.fillPoly(cheek_mask, [cheek_points], 1.0)  # Fill the cheek area

                    # Calculate distance from center for each pixel in cheek region
                    Y, X = np.ogrid[:h, :w]
                    distances = np.sqrt((X - cheek_center[0]) ** 2 + (Y - cheek_center[1]) ** 2)
                    gradient_alpha = alpha * (1 - (distances / max_distance))
                    gradient_alpha = np.clip(gradient_alpha, 0, alpha)  # Limit alpha range

                    # Apply gradient to color and add to main mask
                    for i in range(3):  # Apply color gradient on each channel
                        mask[:, :, i] = np.where(cheek_mask, color[i] * gradient_alpha, mask[:, :, i])

                # Apply blush to each cheek
                create_gradient_blush(left_cheek_indices)
                create_gradient_blush(right_cheek_indices)

            # Apply a slight blur to soften the edges
            blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

            # Blend the blurred mask with the original frame
            frame = cv2.addWeighted(blurred_mask, 1, frame, 1 - alpha, 0)

        return frame


    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_face_landmarks:
            for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
                left_eye_landmarks = [face_landmarks.landmark[idx] for idx in self.LEFT_EYE_INDEXES]
                upper_left_eye_coordinates = self.get_upper_side_coordinates(left_eye_landmarks)
                right_eye_landmarks = [face_landmarks.landmark[idx] for idx in self.RIGHT_EYE_INDEXES]
                
                upper_right_eye_coordinates = self.get_upper_side_coordinates(right_eye_landmarks)
                
               
                Lower_left_eye_coordinates = self.get_lower_side_coordinates(left_eye_landmarks)
                Lower_right_eye_coordinates=self.get_lower_side_coordinates(right_eye_landmarks)

                frame = self.draw_eyeliner(frame, upper_left_eye_coordinates)
                frame = self.draw_eyeliner(frame, upper_right_eye_coordinates)
                frame = self.draw_eyeliner(frame, Lower_left_eye_coordinates)
                frame = self.draw_eyeliner(frame,  Lower_right_eye_coordinates)

                frame = self.apply_lipstick(frame, face_landmarks.landmark, self.LIPS_INDEXES, (0, 0, 255))
                left_cheek_indices = [266,416,345,340]
                right_cheek_indices = [214,36,111,116]
                
                frame = self.apply_blush(frame, left_cheek_indices, right_cheek_indices)

                #changes temporary testing 
                left_eye_shadow_C = [156,224,223,222,56,190,133,173,157,158,159,160,161,246,33,130,130]
                right_eye_shadow_C = [383,444,443,442,286,414,463,398,384,385,386,387,388,263,359]
                frame = self.apply_eyeshadow(frame,left_eye_shadow_C,right_eye_shadow_C,color = (91,123,195) )
        return frame

    def start_video(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = self.process_frame(frame)

            cv2.imshow('Makeup Application', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    makeup_app = MakeupApplication()
    makeup_app.start_video()
