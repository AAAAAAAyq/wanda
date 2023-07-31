# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process alpaca_clean dataset
def get_alpaca(nsamples, seed, seqlen, tokenizer):
    prompter = Prompter("alpaca")
    cutoff_len = seqlen
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt
    
    # Load train and test datasets
    data = load_dataset('json', data_files='datasets/alpaca_data_cleaned.json')
    
    train_val = data["train"].train_test_split(
            train_size=nsamples, shuffle=False, seed=seed
        )
    
    trainenc = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for i in range(nsamples):
        inp = torch.tensor(trainenc['input_ids'][i])
        tar = torch.tensor(trainenc['labels'][i])
        tar[:-1] = -100
        trainloader.append((inp.unsqueeze(0), tar.unsqueeze(0)))
    return trainloader, None

# Load and process wikitext2 dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('text', data_files='datasets/wikitext/wiki.train.raw', split="train")
    testdata = load_dataset('text', data_files='datasets/wikitext/wiki.test.raw', split="train")
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('json', data_files={'train': 'datasets/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('json', data_files={'validation': 'datasets/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc  # [(torch.Size([1, 2048]), torch.Size([1, 2048]))]

def get_german(nsamples, seed, seqlen, tokenizer):
    text = "Origami, auch bekannt als die Kunst des Papierfaltens, ist die Kunst, verschiedene Formen und Muster durch Falten von Papier zu schaffen. Origami hat eine lange Geschichte, die auf verschiedene kulturelle Traditionen zurückgeht, und hat sich weltweit zu einem beliebten Handwerk entwickelt. Im Folgenden finden Sie eine Einführung in die Geschichte, die Arten und die grundlegenden Techniken des Origami. \n\n I. Geschichte des Origami \n1. Ursprung: Origami entstand in China um das 1. Jahrhundert nach Christus mit der Erfindung des Papiers. Ursprünglich wurde Origami hauptsächlich für religiöse Zeremonien und symbolische Geschenke verwendet. \n\n2. Verbreitung: Origami verbreitete sich dann in Japan, wo Mönche es für religiöse Zeremonien verwendeten. In Japan ist Origami als Origami bekannt, was gefaltetes Papier bedeutet. Im Laufe der Zeit hat sich Origami in Japan zu einer einzigartigen Kunstform entwickelt. \n\n3. Globale Verbreitung: Seit Beginn des 20. Jahrhunderts hat sich Origami über die ganze Welt verbreitet. Heute ist Origami zu einem internationalen Kunsthandwerk geworden, das zahllose Liebhaber und Künstler anzieht. \n\nII. Arten von Origami \n1. Traditionelles Origami: Traditionelles Origami ist die älteste und einfachste Form des Origami und zeigt in der Regel einfache Bilder von Tieren und Pflanzen. Beispiele sind die japanischen Tausend-Papier-Kraniche, Frösche und Lotosblumen. \n\n2. Modernes Origami: Moderne Origami-Künstler erforschen immer wieder neue Techniken und Materialien, um Origami-Kunstwerke komplexer und kunstvoller zu gestalten. Modernes Origami umfasst eine Vielzahl von Stilen wie Abstraktion, Kubismus und Surrealismus. \n\n3. Mathematisches Origami: Mathematisches Origami ist eine Kombination aus Origami und Mathematik, bei der Geometrie, Topologie und andere Bereiche der Mathematik durch Origami erforscht werden. Mathematische Origami-Kunstwerke sind oft sehr symmetrisch und ästhetisch ansprechend. \n\n4. Origami-Modelle: Origami-Modelle sind verschiedene dreidimensionale Modelle wie Gebäude, Autos und Roboter, die mit Origami-Techniken hergestellt werden. Sie erfordern in der Regel feine Berechnungen und großes Geschick. \n\5. Biologisches Origami: Biologisches Origami ist die Kunst des Papierfaltens, die die Formen von Lebewesen in der Natur nachahmt. Biologische Origami-Kunstwerke sind in der Regel sehr realistisch, z. B. Insekten, Vögel und Säugetiere. \n\nIII. Grundlegende Origami-Fertigkeiten\n1. Grundlegende Falttechniken: Zu den grundlegenden Origami-Fertigkeiten gehören verschiedene Falttechniken wie Bergfaltung, Talfaltung, Innenfaltung und Außenfaltung. Die Beherrschung dieser grundlegenden Falttechniken ist eine Voraussetzung für die Gestaltung von Origami. \n\n2. Papierauswahl: Das Papier für Origami-Kreationen sollte eine bestimmte Stärke und Festigkeit aufweisen, damit es während des Faltvorgangs seine Form behält. Auch die Farbe und die Beschaffenheit des Papiers beeinflussen die Gesamtwirkung des Werks. \n\n3. Origami-Werkzeuge: Obwohl Origami hauptsächlich von Hand gemacht wird, werden manchmal Werkzeuge benötigt. Verwenden Sie z. B. einen Knochenfalter (Bruchwerkzeug), um Falten zu erzeugen, eine Schere, um das Papier zuzuschneiden, und die Spitze eines Stifts, um Details auf das gefaltete Papier zu zeichnen. \n\n4. Origami-Anleitungen: Für Anfänger ist es hilfreich, eine Origami-Anleitung zum Lernen und Üben zu verwenden. Viele Origami-Tutorials zeigen die Schritte des Origami in illustrierter Form oder als Video, was leicht zu verstehen und nachzuvollziehen ist. \n\n5. Origami-Design: Wenn du die Kunst des Origami beherrschst, kannst du dich an der Gestaltung deiner eigenen Origami-Kreationen versuchen. Das Origamidesign erfordert ein tiefes Verständnis der Natur des Papiers, der Faltmethoden und der räumlichen Komposition. Darüber hinaus sind auch Inspiration und Kreativität wichtige Faktoren beim Origami-Design. \n\n Zusammenfassend lässt sich sagen, dass Origami eine kreative und unterhaltsame Kunstform für Menschen jeden Alters ist. Wenn man die grundlegenden Fertigkeiten und Methoden des Origami beherrscht, kann man in seiner Freizeit den Spaß an der Kunstfertigkeit erleben und gleichzeitig Geduld und genaue Beobachtung entwickeln. Darüber hinaus kann Origami als einzigartiges Geschenk verwendet werden, um Gedanken und Gefühle zu vermitteln. Ich hoffe, dieser Artikel hat Ihnen geholfen, die Geschichte, die Arten und die grundlegenden Techniken des Origami zu verstehen, und ich wünsche Ihnen, dass Sie endlosen Spaß und Erfüllung in der Welt des Origami finden. \n\nAußerdem hat Origami einen hohen Stellenwert im Bereich der Bildung. Origami trainiert die handwerklichen Fähigkeiten, das räumliche Vorstellungsvermögen und die Kreativität der Kinder und vermittelt einige grundlegende mathematische und geometrische Konzepte. Origami kann auch die Kommunikation und Zusammenarbeit bei Teamaktivitäten verbessern. \n\nDie Origami-Kunst wird sich auch in Zukunft weiterentwickeln und ausbauen lassen. So erforschen Forscher beispielsweise, wie Origami-Techniken in Wissenschaft, Technik und Architektur eingesetzt werden können. Origami-Strukturen haben ein breites Spektrum an Anwendungen in Bereichen wie flexible Elektronik, tragbare Geräte und Mikrorobotik. \n\n Auch im Alltag können wir kreativ sein und Origami in Wohndekorationen, Geschenkverpackungen und persönliche Accessoires einbauen. Aus Origami-Blütenblättern lassen sich zum Beispiel Kränze oder Broschen basteln; Origami-Schachteln können zur Aufbewahrung kleiner Gegenstände verwendet werden; Origami-Lesezeichen machen das Lesen zu einem Vergnügen. \Wenn Sie sich für Origami interessieren, sollten Sie sich einigen Origami-Gemeinschaften und -Aktivitäten anschließen. Durch Kommunikation und Lernen können Sie Ihre Erfahrungen mit anderen Origami-Enthusiasten teilen und weitere Origami-Möglichkeiten entdecken. Origami-Wettbewerbe und -Ausstellungen sind ebenfalls eine gute Gelegenheit, um Ihre persönliche Arbeit zu präsentieren und Ihre Fähigkeiten zu verbessern. \n\n Auf dem Weg des Origami können Sie auf einige Rückschläge und Schwierigkeiten stoßen. Solange du jedoch Geduld und Ausdauer bewahrst, es immer wieder versuchst und dich verbesserst, wird die Welt des Origami unendlich viel Spaß und Herausforderungen bieten. Schließlich wirst du dein eigenes wunderschönes Werk erschaffen und den Erfolg und die Freude spüren, die die Origami-Kunst mit sich bringt.\n\nDie Origami-Kunst wird sich auch in Zukunft weiterentwickeln und ausbauen lassen. So erforschen Forscher beispielsweise, wie Origami-Techniken in Wissenschaft, Technik und Architektur eingesetzt werden können. Origami-Strukturen haben ein breites Spektrum an Anwendungen in Bereichen wie flexible Elektronik, tragbare Geräte und Mikrorobotik. \n\n Auch im Alltag können wir kreativ sein und Origami in Wohndekorationen, Geschenkverpackungen und persönliche Accessoires einbauen. Aus Origami-Blütenblättern lassen sich zum Beispiel Kränze oder Broschen basteln; Origami-Schachteln können zur Aufbewahrung kleiner Gegenstände verwendet werden; Origami-Lesezeichen machen das Lesen zu einem Vergnügen. \Wenn Sie sich für Origami interessieren, sollten Sie sich einigen Origami-Gemeinschaften und -Aktivitäten anschließen. Durch Kommunikation und Lernen können Sie Ihre Erfahrungen mit anderen Origami-Enthusiasten teilen und weitere Origami-Möglichkeiten entdecken. Origami-Wettbewerbe und -Ausstellungen sind ebenfalls eine gute Gelegenheit, um Ihre persönliche Arbeit zu präsentieren und Ihre Fähigkeiten zu verbessern. \n\n Auf dem Weg des Origami können Sie auf einige Rückschläge und Schwierigkeiten stoßen. Solange du jedoch Geduld und Ausdauer bewahrst, es immer wieder versuchst und dich verbesserst, wird die Welt des Origami unendlich viel Spaß und Herausforderungen bieten. Schließlich wirst du dein eigenes wunderschönes Werk erschaffen und den Erfolg und die Freude spüren, die die Origami-Kunst mit sich bringt.\n\nDie Origami-Kunst wird sich auch in Zukunft weiterentwickeln und ausbauen lassen. So erforschen Forscher beispielsweise, wie Origami-Techniken in Wissenschaft, Technik und Architektur eingesetzt werden können. Origami-Strukturen haben ein breites Spektrum an Anwendungen in Bereichen wie flexible Elektronik, tragbare Geräte und Mikrorobotik. \n\n Auch im Alltag können wir kreativ sein und Origami in Wohndekorationen, Geschenkverpackungen und persönliche Accessoires einbauen. Aus Origami-Blütenblättern lassen sich zum Beispiel Kränze oder Broschen basteln; Origami-Schachteln können zur Aufbewahrung kleiner Gegenstände verwendet werden; Origami-Lesezeichen machen das Lesen zu einem Vergnügen. \Wenn Sie sich für Origami interessieren, sollten Sie sich einigen Origami-Gemeinschaften und -Aktivitäten anschließen. Durch Kommunikation und Lernen können Sie Ihre Erfahrungen mit anderen Origami-Enthusiasten teilen und weitere Origami-Möglichkeiten entdecken. Origami-Wettbewerbe und -Ausstellungen sind ebenfalls eine gute Gelegenheit, um Ihre persönliche Arbeit zu präsentieren und Ihre Fähigkeiten zu verbessern. \n\n Auf dem Weg des Origami können Sie auf einige Rückschläge und Schwierigkeiten stoßen. Solange du jedoch Geduld und Ausdauer bewahrst, es immer wieder versuchst und dich verbesserst, wird die Welt des Origami unendlich viel Spaß und Herausforderungen bieten. Schließlich wirst du dein eigenes wunderschönes Werk erschaffen und den Erfolg und die Freude spüren, die die Origami-Kunst mit sich bringt."
    
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    trainenc = tokenizer(text, return_tensors='pt')
    i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    j = i + seqlen
    inp = trainenc.input_ids[:, i:j]
    tar = inp.clone()
    tar[:, :-1] = -100
    trainloader.append((inp, tar))
    
    return trainloader, None


def get_chinese(nsamples, seed, seqlen, tokenizer):
    text = "折纸，又称为折纸艺术，是一种通过折叠纸张来创作各种形状和图案的艺术。折纸历史悠久，源于不同的文化传统，现已成为全球范围内的流行手工艺。以下是关于折纸的历史、种类和基本技巧的介绍。\n\n一、折纸历史\n1. 起源：折纸起源于中国，约在公元1世纪时，随着纸张的发明而诞生。最初，折纸主要用于宗教仪式和象征性的礼物。\n\n2. 传播：折纸随后传播到日本，那里的僧侣将其应用于宗教仪式。在日本，折纸被称为“折り紙”（origami），意为“折纸”。随着时间的推移，折纸在日本逐渐发展成为一种独特的艺术形式。\n\n3. 全球范围：从20世纪初开始，折纸逐渐传播到世界各地。现在，折纸已成为一种国际性的手工艺，吸引着无数爱好者和艺术家。\n\n二、折纸种类\n1. 传统折纸：传统折纸是最古老和最基本的折纸形式，通常以简单的动物和植物形象为主题。例如，日本的千纸鹤、青蛙和莲花等。\n\n2. 现代折纸：现代折纸艺术家们不断探索新的技巧和材料，使折纸作品更加复杂和精细。现代折纸包括多种风格，如抽象派、立体主义和超现实主义等。\n\n3. 数学折纸：数学折纸是折纸与数学的结合，通过折纸探讨几何学、拓扑学和其他数学领域的问题。数学折纸作品往往具有高度的对称性和美学价值。\n\n4. 折纸模型：折纸模型是指用折纸技巧制作的各种立体模型，如建筑、汽车和机器人等。这些作品通常需要精细的计算和高超的技巧。\n\n5. 生物折纸：生物折纸是一种模仿自然界生物形态的折纸艺术。生物折纸作品通常具有高度的逼真度，如昆虫、鸟类和哺乳动物等。\n\n三、折纸基本技巧\n1. 基本折法：折纸的基本技巧包括各种折法，如山折、谷折、内翻折、外翻折等。熟练掌握这些基本折法是进行折纸创作的前提。\n\n2. 纸张选择：折纸作品的纸张应具有一定的厚度和强度，以便在折叠过程中保持形状。同时，纸张的颜色和质地也会影响作品的整体效果。\n\n3. 折纸工具：虽然折纸主要依靠手工操作，但有时也需要借助工具。例如，使用骨折工具（bone folder）可以使折痕更加清晰；使用剪刀可以对纸张进行修整；使用笔尖可以在折纸上绘制细节。\n\n4. 折纸教程：对于初学者来说，按照折纸教程进行学习和练习是非常有帮助的。许多折纸教程以图文或视频形式展示折纸步骤，易于理解和操作。\n\n5. 折纸设计：熟练掌握折纸技巧后，可以尝试自己设计折纸作品。折纸设计需要对纸张的性质、折叠方法和空间构成有深刻的理解。此外，灵感和创意也是折纸设计的关键因素。\n\n总之，折纸是一种富有创意和趣味的艺术形式，适合各个年龄段的人群。掌握折纸的基本技巧和方法，可以让我们在闲暇之余体验到手工艺的乐趣，同时培养耐心和细致观察力。此外，折纸还可以作为一种独特的礼物，传递心意和情感。希望这篇文章能对你了解折纸的历史、种类和基本技巧有所帮助，祝你在折纸的世界里找到无尽的乐趣和成就感。\n\n此外，折纸在教育领域也具有很高的价值。折纸能够锻炼孩子们的动手能力、空间想象力和创造力，同时还可以教授一些基本的数学和几何概念。在团队活动中，折纸还可以增进沟通和合作能力。\n\n未来，折纸艺术可能会继续发展和演变。例如，研究人员正在探索如何将折纸技术应用于科学、工程和建筑领域。折纸结构在柔性电子、可穿戴设备和微型机器人等领域具有广泛的应用前景。\n\n在日常生活中，我们也可以发挥创意，将折纸融入家居装饰、礼品包装和个人饰品等方面。例如，折纸花瓣可以用于制作花环或胸针；折纸盒子可以用于收纳小物件；折纸书签可以为阅读增添一份趣味。\n\n如果你对折纸产生了兴趣，不妨尝试参加一些折纸社群和活动。通过交流和学习，你可以与其他折纸爱好者分享经验，发现更多折纸的可能性。同时，折纸比赛和展览也是展示个人作品、提高技艺的好机会。\n\n在折纸的道路上，你可能会遇到一些挫折和困难。然而，只要保持耐心和毅力，不断尝试和改进，你会发现折纸的世界充满无限的乐趣和挑战。最终，你将创作出属于自己的精美作品，感受到折纸艺术带来的成就和喜悦。"
    
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    
    trainenc = tokenizer(text, return_tensors='pt')
    i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    j = i + seqlen
    inp = trainenc.input_ids[:, i:j]
    tar = inp.clone()
    tar[:, :-1] = -100
    trainloader.append((inp, tar))
    
    return trainloader, None
    

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "chinese" in name:
        return get_chinese(nsamples, seed, seqlen, tokenizer)
    elif "german" in name:
        return get_german(nsamples, seed, seqlen, tokenizer)
    elif "alpaca" in name:
        return get_alpaca(nsamples, seed, seqlen, tokenizer)
    elif "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    
if __name__ == "__main__": 
    get_loaders('wikitext2', seed=0, seqlen=2048, tokenizer=None)