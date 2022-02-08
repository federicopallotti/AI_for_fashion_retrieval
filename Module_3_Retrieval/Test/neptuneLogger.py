import neptune.new as neptune

class NetpuneLogger:

    def __init__(self, category, train_params, hyper):
        self.run = neptune.init(project='apalese/TestingNNRetrieval',
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YmE0NjU5Ni1lZDk4LTRkMmEtYjIwYS02YjM5NGJmYWNlMmQifQ==',
                           )


        self.run["Targets"] = f"{category}"
        self.run["JIRA"] = "NPT-952"
        self.run["algorithm"] = "TripletLossTest"


        self.run["parameters"] = hyper
        self.run["train parameters"] = train_params

        self.recalls = {1: 'test/Recall_1', 5: 'test/Recall_5', 10: 'test/Recall_10', 20: 'test/Recall_20'}
        self.update_recalls = {1: 'test/score1', 5: 'test/score5', 10: 'test/score10', 20: 'test/score20'}

    def set(self, score, n, r):
        self.run[self.recalls[r]].log(score / n)

    def setFailure(self, len):
        self.run[f"test/Recall_20"].log(0)

    def update(self, score, r):
        self.run[self.update_recalls[r]] = score

    def destroy(self):
        self.run.stop()
