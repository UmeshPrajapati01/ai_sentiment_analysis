import os

def analyze_image_classes(base_path, output_file):
    """
    Analyze `cat_classifieddataimage` folder and write a text summary to `output_file`.
    """
    images_path = os.path.join(base_path, 'colected_data', 'imagefiles', 'cat_classifieddataimage')

    if not os.path.exists(images_path):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Error: Path {images_path} does not exist.\n")
        return

    try:
        classes = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
        total_classes = len(classes)
        total_files = 0
        all_extensions = set()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Total classes: {total_classes}\n")
            f.write(f"Classes: {', '.join(sorted(classes))}\n")
            f.write("\nClass details:\n")

            for cls in sorted(classes):
                cls_path = os.path.join(images_path, cls)
                files = [fn for fn in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, fn))]
                num_files = len(files)
                total_files += num_files

                class_extensions = set()
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext:
                        class_extensions.add(ext)
                        all_extensions.add(ext)

                f.write(f"- {cls}: {num_files} image files, extensions: {', '.join(sorted(class_extensions)) if class_extensions else 'None'}\n")

            f.write(f"\nTotal image files across all classes: {total_files}\n")
            f.write(f"Image file extensions (all): {', '.join(sorted(all_extensions)) if all_extensions else 'None'}\n")
    except Exception as e:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Error during analysis: {e}\n")


if __name__ == '__main__':
    # Determine project root relative to this script
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_path, 'colected_data', 'cleaned_data', 'cleaned_imagedata')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'image_analysis.txt')
    analyze_image_classes(base_path, output_file)
    print(f"Analysis output saved to: {output_file}")
