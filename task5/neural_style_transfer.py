import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def load_image(img_path , max_size=512):
    image = Image.open(img_path).convert("RGB")
    size = min(max(image.size), max_size)#so taking the min between the max size of the image or the max size we define
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])


    image =  transform(image).unsqueeze(0) # adds batch (WHB) to (1WHB) vimp for image to work with models
    return image

content_img = load_image("mountains.jpg").to(device)
style_img = load_image("watercolor-painting-sunset-with-mountain-background_791234-1077-197762370.jpg").to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_image = content_img.to(device)
style_image = style_img.to(device)

vgg = models.vgg19(pretrained=True).features.to(device).eval()
content_layers = ["21"]
style_layers = ['0', '5', '10', '19', '28']

#Extracting the features of the img
def get_features(image, model):
    features = {}
    x = image
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if str(name) in content_layers + style_layers:
            features[str(name)] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram


def style_loss(target_grams, style_grams):
    loss = 0
    for layer in style_layers:
        loss += torch.mean((target_grams[layer] - style_grams[layer]) ** 2)
    return loss


def content_loss(target_features, content_features):
    return torch.mean((target_features["21"] - content_features["21"]) ** 2)


target_image = content_image.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target_image], lr=0.002)
style_weight = 1e5
content_weight = 1e1

def show_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()



content_features = get_features(content_image, vgg)
style_features = get_features(style_image, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}
epochs = 500
for i in range(epochs):
    target_features = get_features(target_image , vgg)
    target_grams = {layer: gram_matrix(target_features[layer]) for layer in style_layers}

    s_loss = style_loss(target_grams , style_grams)
    c_loss = content_loss(target_features , content_features)

    loss = style_weight * s_loss + content_weight * c_loss

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()


    print(f"Epoch [{i}/{epochs}] Content loss : {c_loss.item():.4f} Style Loss {s_loss.item():.4f}")





show_image(target_image)
target_image = target_image.cpu().clone().detach().squeeze(0)
transforms.ToPILImage()(target_image).save('stylized_image2.jpg')











