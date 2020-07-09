from captcha.image import ImageCaptcha
from config import char_set
import random


def generate_random_text():
    text = ""
    for i in range(4):
        text += random.choice(char_set)
    return text


if __name__ == '__main__':
    N = 10000
    for i in range(N):
        captchaGenerator = ImageCaptcha(
            width=random.randint(100, 200),
            height=random.randint(40, 80),
            fonts=['/System/Library/Fonts/Supplemental/Courier New.ttf',
                   '/System/Library/Fonts/Supplemental/Courier New Bold Italic.ttf',
                   '/System/Library/Fonts/Supplemental/Courier New Bold.ttf',
                   '/System/Library/Fonts/Supplemental/Courier New Italic.ttf',

                   '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                   '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf',
                   '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                   '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf',

                   '/System/Library/Fonts/Supplemental/Georgia Bold.ttf',
                   '/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf',
                   '/System/Library/Fonts/Supplemental/Georgia Bold.ttf',
                   '/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf',
                   ]
        )
        text = generate_random_text()
        captchaGenerator.write(text, 'samples/py_captcha/{}_{}_0.png'.format(text, i))

        if i % 100 == 99:
            print('%03f%%' % ((i + 1)/N * 100))

