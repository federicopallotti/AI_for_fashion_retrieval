import neptune.new as neptune
import time

class NetpuneLogger:

    def __init__(self):
        self.run = neptune.init(project='apalese/TestYOOX',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YmE0NjU5Ni1lZDk4LTRkMmEtYjIwYS02YjM5NGJmYWNlMmQifQ==')


    def imsave(self, path, img):
        self.run[str(path)].upload(path)
    def destroy(self):
        self.run.stop()
