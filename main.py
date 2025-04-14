from harris_stitcher import *
import matplotlib.pyplot as plt
import os

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

        stitched = stitch_images(im1, im2, translation)

        plt.imshow(stitched, cmap='gray')
        plt.title(f"Stitched: {name1} + {name2}")
        plt.axis('off')
        plt.show()

        visualize_matches(im1, im2, coords1, coords2, matches)

if __name__ == "__main__":
    main()
