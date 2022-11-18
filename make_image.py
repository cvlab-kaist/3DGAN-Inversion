from PIL import Image, ImageDraw, ImageFont

width = 512 * 3
height = 100
message = 'Pre-trained latent code encoder and camera pose estimator provide good initializations for optimization.'
# message = 'Our depth-based warping scheme provides a more accurate and stable optimization.'
# message = 'Novel view synthesis using learned latent code and camera pose.'
# message = 'We boost the reconstruction ability using the popular pivotal tuning method.'
# message = "Using depth regularizaiton during pivotal tuning enables novel view synthesis with a more stable geometry."

font = ImageFont.truetype("Economica-Regular.ttf", size=40)

img = Image.new('RGB', (width, height), color='white')

imgDraw = ImageDraw.Draw(img)

textWidth, textHeight = imgDraw.textsize(message, font=font)
xText = (width - textWidth) / 2
yText = (height - textHeight) / 2

imgDraw.text((xText, yText), message, font=font, fill=(0, 0, 0))

img.save('prompt0.png')