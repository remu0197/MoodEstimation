import glob, csv, sys, os
import MeCab
import codecs

class TextToMorpheme:
    def __init__(self):
        self.__category_list = []

        with codecs.open("../data/pos-id.def") as f:
            lines = f.readlines()
            for line in lines:
                morph_conc = ""
                morph_base = line.split(" ")[0]
                morphes = morph_base.split(",")

                for morph in morphes:
                    if morph != "*":
                        morph_conc += morph + "-"

                morph_conc = morph_conc.rstrip("-")
                self.__category_list.append(morph_conc)

    def __get_morph_id(self, category):
        for i, _category in enumerate(self.__category_list):
            if category == _category:
                return i

        return 0

    def to_morpheme(self, text, mode="-Ochasen", is_id=True):
        m = MeCab.Tagger(mode)
        parse = m.parse(text)
        parse_split = parse.split("\t")
        OUTPUT_COUNT = 5

        morph_list = []
        for i in range(int(len(parse_split) / OUTPUT_COUNT)):
            morph = parse_split[i*OUTPUT_COUNT]
            if "\n" in morph:
                index = morph.find("\n") + 1
                morph = morph[index:]
            category = parse_split[i*OUTPUT_COUNT + 3]
            id = self.__get_morph_id(category)

            if is_id:
                morph_list.append([morph, id])
            else:
                morph_list.append([morph, category])

        return morph_list

    def from_csv(self, csv_filepath):
        morph_lists = []
        with open(csv_filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                morph_lists.append(self.to_morpheme(text=row[2]))

        return morph_lists

def filler_checker(is_update=False):
    except_id_list = []
    except_id_list.extend(range(10))
    except_id_list.extend(range(13,31))

    INPUT_DIR = "../data/edit_texts/"
    OUTPUT_DIR = "../data/except_sections/"
    csv_pathes = glob.glob(INPUT_DIR + "/**/*.csv")

    TTM = TextToMorpheme()
    for path in csv_pathes:
        subject_info = path.lstrip(OUTPUT_DIR).lstrip("\\")
        filepath = subject_info.split("\\")
        subject_group, filename = filepath[0], filepath[1]

        output_path = OUTPUT_DIR + subject_group + "/" + filename
        if os.path.exists(output_path) and not is_update:
            print("SKIP   :" + path)
            continue

        if not os.path.exists(OUTPUT_DIR + subject_group):
            os.mkdir(OUTPUT_DIR + subject_group)

        print("TARGET   :" + path)

        check_list = []
        morph_lists = TTM.from_csv(path)
        
        for morph_list in morph_lists:
            text = ""
            is_filler = "1"

            for morph in morph_list:
                text += morph[0]
                if not morph[1] in except_id_list:
                    is_filler = "0"
            
            check_list.append([is_filler, text])

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(check_list)

if __name__ == "__main__":
    # filler_checker()
    TTM = TextToMorpheme()
    print(TTM.to_morpheme("ほう", is_id=False))