from PIL import Image
import matplotlib.pyplot as plt

content_img = Image.open('mountains.jpg')
style_img = Image.open('watercolor-painting-sunset-with-mountain-background_791234-1077-197762370.jpg')
stylized_img = Image.open('stylized_image2.jpg')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(content_img)
plt.title('Content Image : Mountains')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(style_img)
plt.title('Style Image : WaterColour')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(stylized_img)
plt.title('The Result')
plt.axis('off')

plt.tight_layout()
plt.show()
