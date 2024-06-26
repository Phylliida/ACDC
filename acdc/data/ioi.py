from .utils import restrict_to_most_common_size
import re
import random

# modified from https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/ioi/ioi_dataset.py


ABC_TEMPLATES = [
    "Then, [NAME], [NAME] and [NAME] went to the [PLACE]. [NAME] and [NAME] gave a [OBJECT] to",
    "Afterwards [NAME], [NAME] and [NAME] went to the [PLACE]. [NAME] and [NAME] gave a [OBJECT] to",
    "When [NAME], [NAME] and [NAME] arrived at the [PLACE], [NAME] and [NAME] gave a [OBJECT] to",
    "Friends [NAME], [NAME] and [NAME] went to the [PLACE]. [NAME] and [NAME] gave a [OBJECT] to",
]

BABA_TEMPLATES = [
    "Then, [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then, [NAME] and [NAME] had a lot of fun at the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then, [NAME] and [NAME] were working at the [PLACE]. [NAME] decided to give a [OBJECT] to",
    "Then, [NAME] and [NAME] were thinking about going to the [PLACE]. [NAME] wanted to give a [OBJECT] to",
    "Then, [NAME] and [NAME] had a long argument, and afterwards [NAME] said to",
    "After [NAME] and [NAME] went to the [PLACE], [NAME] gave a [OBJECT] to",
    "When [NAME] and [NAME] got a [OBJECT] at the [PLACE], [NAME] decided to give it to",
    "When [NAME] and [NAME] got a [OBJECT] at the [PLACE], [NAME] decided to give the [OBJECT] to",
    "While [NAME] and [NAME] were working at the [PLACE], [NAME] gave a [OBJECT] to",
    "While [NAME] and [NAME] were commuting to the [PLACE], [NAME] gave a [OBJECT] to",
    "After the lunch, [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Afterwards, [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then, [NAME] and [NAME] had a long argument. Afterwards [NAME] said to",
    "The [PLACE] [NAME] and [NAME] went to had a [OBJECT]. [NAME] gave it to",
    "Friends [NAME] and [NAME] found a [OBJECT] at the [PLACE]. [NAME] gave it to",
]

BABA_LONG_TEMPLATES = [
    "Then in the morning, [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then in the morning, [NAME] and [NAME] had a lot of fun at the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then in the morning, [NAME] and [NAME] were working at the [PLACE]. [NAME] decided to give a [OBJECT] to",
    "Then in the morning, [NAME] and [NAME] were thinking about going to the [PLACE]. [NAME] wanted to give a [OBJECT] to",
    "Then in the morning, [NAME] and [NAME] had a long argument, and afterwards [NAME] said to",
    "After taking a long break [NAME] and [NAME] went to the [PLACE], [NAME] gave a [OBJECT] to",
    "When soon afterwards [NAME] and [NAME] got a [OBJECT] at the [PLACE], [NAME] decided to give it to",
    "When soon afterwards [NAME] and [NAME] got a [OBJECT] at the [PLACE], [NAME] decided to give the [OBJECT] to",
    "While spending time together [NAME] and [NAME] were working at the [PLACE], [NAME] gave a [OBJECT] to",
    "While spending time together [NAME] and [NAME] were commuting to the [PLACE], [NAME] gave a [OBJECT] to",
    "After the lunch in the afternoon, [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Afterwards, while spending time together [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then in the morning afterwards, [NAME] and [NAME] had a long argument. Afterwards [NAME] said to",
    "The local big [PLACE] [NAME] and [NAME] went to had a [OBJECT]. [NAME] gave it to",
    "Friends separated at birth [NAME] and [NAME] found a [OBJECT] at the [PLACE]. [NAME] gave it to",
]

BABA_LATE_IOS = [
    "Then, [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then, [NAME] and [NAME] had a lot of fun at the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then, [NAME] and [NAME] were working at the [PLACE]. [NAME] decided to give a [OBJECT] to",
    "Then, [NAME] and [NAME] were thinking about going to the [PLACE]. [NAME] wanted to give a [OBJECT] to",
    "Then, [NAME] and [NAME] had a long argument and after that [NAME] said to",
    "After the lunch, [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Afterwards, [NAME] and [NAME] went to the [PLACE]. [NAME] gave a [OBJECT] to",
    "Then, [NAME] and [NAME] had a long argument. Afterwards [NAME] said to",
]

BABA_EARLY_IOS = [
    "Then [NAME] and [NAME] went to the [PLACE], and [NAME] gave a [OBJECT] to",
    "Then [NAME] and [NAME] had a lot of fun at the [PLACE], and [NAME] gave a [OBJECT] to",
    "Then [NAME] and [NAME] were working at the [PLACE], and [NAME] decided to give a [OBJECT] to",
    "Then [NAME] and [NAME] were thinking about going to the [PLACE], and [NAME] wanted to give a [OBJECT] to",
    "Then [NAME] and [NAME] had a long argument, and after that [NAME] said to",
    "After the lunch [NAME] and [NAME] went to the [PLACE], and [NAME] gave a [OBJECT] to",
    "Afterwards [NAME] and [NAME] went to the [PLACE], and [NAME] gave a [OBJECT] to",
    "Then [NAME] and [NAME] had a long argument, and afterwards [NAME] said to",
]


PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]

OBJECTS = [
    "apple", # ring confuses the model
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

NOUNS_DICT = {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}

interventions = """
CAB AB C
DAB AB D
-
ACB AB C
ADB AB D
-
ABC AB C
ABD AB D
-
ABC AB C
ABC AC B
-
ABC AC B
ABC BC A
"""

def get_all_single_name_abc_patching_formats():
    lines = [x for x in interventions.split("\n") if len(x) == 8]
    for i in range(0, len(lines), 2):
        abc_format = lines[i] + "\n" + lines[i+1]
        yield abc_format

def ioi_data_generator(tokenizer, num_patching_pairs, templates, patching_formats):
    '''
    templates should be from the templates above
    patching_formats should be a list of things that look like this for three names:

    ABC AB C
    ABC AC B

    gives:
    
    uncorrupted:
    A and B and C had fun at the store. A and B gave a bike to (answer is C)
    corrupted:
    A and B and C had fun at the store. A and C gave a bike to (answer is B)

    and this for two names

    AB A B
    AB B A

    gives:

    uncorrupted:
    A and B had fun at the store. A gave a bike to (answer is B)
    corrupted:
    A and B had fun at the store. B gave a bike to (answer is A)

    a random template and patching format will be chosen for each patching pair
    '''
    global good_names
    global good_nouns
    if not 'good_names' in globals() or good_names is None:
        names = restrict_to_most_common_size(tokenizer, NAMES, with_space=True, force_size=1)
        names = sorted(list(names))

        # stuff like "chris" and "christine" get confused, ignore them
        no_prefix_names = []
        for name in names:
            has_other_as_prefix = any([other.startswith(name) and other != name for other in names])
            if not has_other_as_prefix:
                no_prefix_names.append(name)
            #else:
            #    other_prefix = names[[other.startswith(name) and other != name for other in names].index(True)]
        names = no_prefix_names
        good_names = sorted(names)

        noun_dict = {}
        for k,v in NOUNS_DICT.items():
            noun_dict[k] = sorted(restrict_to_most_common_size(tokenizer, v, with_space=True))
        good_nouns = noun_dict
        
    good_names = sorted(good_names)
    for n in range(num_patching_pairs):
        template = random.choice(templates) 
        patching_format = random.choice(patching_formats)   
        # sorted is important for determinism
        unique_tokens = sorted(list(set(re.sub(r"\s*", "", patching_format))))
        random.shuffle(good_names)
        tok_map = {}
        for ind, tok in enumerate(unique_tokens):
            tok_map[tok] = good_names[ind]
        place = random.choice(good_nouns['[PLACE]'])
        object = random.choice(good_nouns['[OBJECT]'])
        def insert_entities(text, format):
            for s in format:
                # replace the first [NAME] with the given name
                # this will work its way through the whole template
                text = text.replace("[NAME]", tok_map[s], 1)
            text = text.replace("[PLACE]", place)
            text = text.replace("[OBJECT]", object)
            return text
        prompts = []
        answers = []
        for line in patching_format.split("\n"):
            line = line.replace(" ", "")
            if len(line) > 0:
                answer = line[-1] # last is the answer
                answers.append(" " + tok_map[answer])

                # rest are the format
                format = line[:-1]
                prompt = insert_entities(text=template, format=format)
                prompts.append(prompt) 
        all_answers = set([" " + name for name in tok_map.values()])
        for prompt, answer in zip(prompts, answers):
            correct = [answer]
            # all possible other answers are the incorrect options
            incorrect = sorted(list(all_answers - set([answer])))
            yield prompt, correct, incorrect

# names from https://www.ssa.gov/oact/babynames/decades/names2000s.html
NAMES = ["Jacob",
"Emily",
"Michael",
"Madison",
"Joshua",
"Emma",
"Matthew",
"Olivia",
"Daniel",
"Hannah",
"Christopher",
"Abigail",
"Andrew",
"Isabella",
"Ethan",
"Samantha",
"Joseph",
"Elizabeth",
"William",
"Ashley",
"Anthony",
"Alexis",
"David",
"Sarah",
"Alexander",
"Sophia",
"Nicholas",
"Alyssa",
"Ryan",
"Grace",
"Tyler",
"Ava",
"James",
"Taylor",
"John",
"Brianna",
"Jonathan",
"Lauren",
"Noah",
"Chloe",
"Brandon",
"Natalie",
"Christian",
"Kayla",
"Dylan",
"Jessica",
"Samuel",
"Anna",
"Benjamin",
"Victoria",
"Nathan",
"Mia",
"Zachary",
"Hailey",
"Logan",
"Justin",
"Jasmine",
"Gabriel",
"Julia",
"Jose",
"Morgan",
"Austin",
"Destiny",
"Kevin",
"Rachel",
"Elijah",
"Ella",
"Caleb",
"Kaitlyn",
"Robert",
"Megan",
"Thomas",
"Katherine",
"Jordan",
"Savannah",
"Cameron",
"Jennifer",
"Jack",
"Alexandra",
"Hunter",
"Allison",
"Jackson",
"Haley",
"Maria",
"Isaiah",
"Kaylee",
"Evan",
"Lily",
"Isaac",
"Makayla",
"Luke",
"Brooke",
"Mason",
"Nicole",
"Jayden",
"Mackenzie",
"Jason",
"Addison",
"Gavin",
"Stephanie",
"Aaron",
"Lillian",
"Connor",
"Andrea",
"Aiden",
"Faith",
"Aidan",
"Zoe",
"Kyle",
"Kimberly",
"Juan",
"Madeline",
"Charles",
"Alexa",
"Luis",
"Katelyn",
"Adam",
"Gabriella",
"Lucas",
"Gabrielle",
"Brian",
"Eric",
"Amanda",
"Adrian",
"Kylie",
"Nathaniel",
"Mary",
"Sean",
"Paige",
"Alex",
"Riley",
"Carlos",
"Leah",
"Bryan",
"Jenna",
"Ian",
"Sara",
"Owen",
"Rebecca",
#"Jesus", Jesus is too powerful
"Michelle",
"Landon",
"Sofia",
"Julian",
"Vanessa",
"Chase",
"Jordan",
"Cole",
"Angelina",
"Caroline",
"Jeremiah",
"Avery",
"Steven",
"Audrey",
"Sebastian",
"Evelyn",
"Xavier",
"Maya",
"Timothy",
"Claire",
"Carter",
"Autumn",
"Wyatt",
"Jocelyn",
"Brayden",
"Ariana",
"Blake",
"Nevaeh",
"Hayden",
"Arianna",
"Devin",
"Jada",
"Cody",
"Bailey",
"Richard",
"Seth",
"Aaliyah",
"Dominic",
"Amber",
"Jaden",
"Isabel",
"Antonio",
"Mariah",
"Miguel",
"Danielle",
"Liam",
"Melanie",
"Patrick",
"Sierra",
"Carson",
"Erin",
"Jesse",
"Amelia",
"Tristan",
"Molly",
"Alejandro",
"Isabelle",
"Henry",
"Melissa",
"Victor",
"Madelyn",
"Trevor",
"Jacqueline",
"Bryce",
"Marissa",
"Jake",
"Angela",
"Riley",
"Shelby",
"Colin",
"Leslie",
"Jared",
"Katie",
"Jeremy",
"Jade",
"Mark",
"Catherine",
"Caden",
"Diana",
"Garrett",
"Aubrey",
"Parker",
"Mya",
"Marcus",
"Amy",
"Vincent",
"Briana",
"Kaleb",
"Sophie",
"Kaden",
"Gabriela",
"Brady",
"Breanna",
"Colton",
"Gianna",
"Kenneth",
"Kennedy",
"Joel",
"Gracie",
"Oscar",
"Peyton",
"Josiah",
"Adriana",
"Jorge",
"Christina",
"Ashton",
"Courtney",
"Cooper",
"Daniela",
"Tanner",
"Lydia",
"Eduardo",
"Kathryn",
"Paul",
"Valeria",
"Edward",
"Layla",
"Ivan",
"Alexandria",
"Preston",
"Natalia",
"Maxwell",
"Angel",
"Alan",
"Laura",
"Charlotte",
"Stephen",
"Margaret",
"Grant",
"Cheyenne",
"Nicolas",
"Naomi",
"Dakota",
"Miranda",
"Omar",
"Mikayla",
"Alexis",
"Kelsey",
"George",
"Payton",
"Eli",
"Ana",
"Collin",
"Alicia",
"Spencer",
"Jillian",
"Gage",
"Max",
"Mckenzie",
"Ricardo",
"Ashlyn",
"Cristian",
"Sabrina",
"Derek",
"Caitlin",
"Micah",
"Brody",
"Valerie",
"Nolan",
"Rylee",
"Ayden",
"Skylar",
"Dalton",
"Lindsey",
"Shane",
"Kelly",
"Peter",
"Genesis",
"Damian",
"Zoey",
"Jeffrey",
"Eva",
"Brendan",
"Sadie",
"Travis",
"Alexia",
"Fernando",
"Cassidy",
"Peyton",
"Kylee",
"Conner",
"Kendall",
"Andres",
"Jordyn",
"Javier",
"Kate",
"Giovanni",
"Jayla",
"Shawn",
"Karen",
"Braden",
"Tiffany",
"Jonah",
"Cassandra",
"Bradley",
"Juliana",
"Cesar",
"Reagan",
"Emmanuel",
"Caitlyn",
"Manuel",
"Giselle",
"Edgar",
"Serenity",
"Mario",
"Alondra",
"Erik",
"Lucy",
"Edwin",
"Bianca",
"Johnathan",
"Kiara",
"Devon",
"Erick",
"Erica",
"Wesley",
"Angelica",
"Oliver",
"Trenton",
"Chelsea",
"Hector",
"Alana",
"Malachi",
"Liliana",
"Jalen",
"Brittany",
"Raymond",
"Camila",
"Gregory",
"Makenzie",
"Abraham",
"Lilly",
"Elias",
"Veronica",
"Leonardo",
"Abby",
"Sergio",
"Jazmin",
"Donovan",
"Adrianna",
"Colby",
"Delaney",
"Marco",
"Karina",
"Bryson",
"Ellie",
"Martin",
"Jasmin",
]
