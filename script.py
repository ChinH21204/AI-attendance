from PIL import Image

img = Image.open("your_image.jpg")
resized_img = img.resize((800, 600), Image.LANCZOS)  # LANCZOS giữ nét tốt nhất
resized_img.save("resized_image.jpg")
