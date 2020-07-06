from dataset import build_model
from tensorflow.keras import Model
from sample import samples


def train(model: Model):
    model.fit_generator(
        samples("samples/train", 1000),
        steps_per_epoch=90,
        epochs=10)
    return model


def evaluate(model: Model):
    test_loss, test_acc = model.evaluate_generator(
        samples("samples/test", 1000),
        steps=1,
        verbose=1)
    print('accuracy: {}, loss: {}'.format(test_acc, test_loss))


if __name__ == '__main__':
    model = build_model()
    train(model)
    evaluate(model)
    model.save('captcha_model')
