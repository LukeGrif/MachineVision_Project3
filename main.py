"""
main.py

Author: Luke Griffin 21334528, Patrick Crotty 21336113, Michael Cronin 21329001, Aaron Smith 21335168,
    Cullen Toal 21306133
Date: 18-04-2025

Description:
This script runs the full Harris Corner Detection and image stitching pipeline.
"""

from harris_stitcher import *
import matplotlib.pyplot as plt


def main():
    image_pairs = [
        ("image-pairs/arch1.png", "image-pairs/arch2.png"),
        ("image-pairs/balloon1.png", "image-pairs/balloon2.png"),
        ("image-pairs/tigermoth1.png", "image-pairs/tigermoth2.png")
    ]

    for name1, name2 in image_pairs:
        print(f"\nProcessing: {name1} and {name2}")
        im1 = load_image_grayscale(name1)
        im2 = load_image_grayscale(name2)

        h1 = harris_response(im1)
        h2 = harris_response(im2)

        coords1 = get_harris_points(h1)
        coords2 = get_harris_points(h2)

        desc1 = get_descriptors(im1, coords1)
        desc2 = get_descriptors(im2, coords2)

        matches = match_descriptors(desc1, desc2)

        if len(matches) == 0:
            print("No matches found.")
            continue

        translation = exhaustive_ransac(matches, coords1, coords2)
        print("Estimated translation:", translation)

        # Compute stitched image
        stitched = stitch_images(im1, im2, translation)

        # Create a figure with two subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # match lines on side-by-side image
        combined = np.hstack((im1, im2))
        ax1.imshow(combined, cmap='gray')
        ax1.set_title("Visualized Matches")
        for i, j in matches:
            r1, c1 = coords1[i]
            r2, c2 = coords2[j]
            ax1.plot([c1, c2 + im1.shape[1]], [r1, r2], 'r', linewidth=0.5)
        ax1.axis('off')

        # stitched image only
        ax2.imshow(stitched, cmap='gray')
        ax2.set_title(f"Stitched Image: {name1} + {name2}")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
