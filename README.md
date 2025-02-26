# PRODIGY_GA_05
Neural Style Transfer using VGG19


🎨 Neural Style Transfer with VGG19
This repository contains the implementation of Neural Style Transfer (NST) using VGG19 in PyTorch. The model combines the content of one image with the artistic style of another to generate stunning, stylized outputs.

📸 How It Works
Neural Style Transfer works by optimizing an input image to minimize a loss function that balances two key components:

Content Loss: Ensures the generated image retains the core structure and objects from the content image.
Style Loss: Captures the artistic patterns, textures, and colors from the style image using Gram matrices.
The model uses a pre-trained VGG19 network to extract feature maps at different layers, combining content and style representations.

🚀 Features
Load content and style images with preprocessing (resizing, normalization).
Compute content and style loss using VGG19 feature maps.
Optimize the target image with Adam optimizer.
Save and display the final stylized image.
Supports CUDA for faster processing on GPUs.

🏗️ Training
The training loop iteratively updates the target image to reduce the weighted sum of content and style losses:

python
Copy
Edit
loss = style_weight * style_loss + content_weight * content_loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
📚 Results
The output image is a unique blend of the content image’s structure and the style image’s artistic flair.

Here’s an example of the results:

Content Image
Style Image
Generated Stylized Image
