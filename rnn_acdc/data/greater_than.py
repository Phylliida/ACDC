from collections import defaultdict
from .utils import restrict_to_most_common_size
import torch

# modified from ACDC https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/greaterthan/utils.py
NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign", 
    "captivity", "case", "challenge", "chaos", "clash", 
    "collaboration", "coma", "competition", "confrontation", "consequence", 
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal", 
    "decline", "decrease", "demonstrations", "development", "disagreement", 
    "disorder", "dispute", "domination", "dynasty", "effect", 
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion", 
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness", 
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey", 
    "kingdom", "marriage", "modernization", "negotiation",
    "notoriety", "obstruction", "operation", "order", "outbreak", 
    "outcome", "overhaul", "patrols", "pilgrimage", "plague",
    "plan", "practice", "process", "program", "progress", 
    "project", "pursuit", "quest", "raids", "reforms", 
    "reign", "relationship",
    "retaliation", "riot", "rise", "rivalry", "romance", 
    "rule", "sanctions", "shift", "siege", "slump", 
    "stature", "stint", "strikes", "study",
    "test", "testing", "tests", "therapy", "tour", 
    "tradition", "treaty", "trial", "trip", "unemployment", 
    "voyage", "warfare", "work",
]

def greater_than_data_generator(tokenizer, num_patching_pairs):
    YEARS = []
    DECADES = []
    for i in range(100):
        s = str(i)
        if i < 10:
            s = f"0{s}"
        DECADES.append(tokenizer.encode(s)[0])
    DECADES = [(tokenizer.decode(tok), tok) for tok in DECADES]
    MINIMUMS = {}

    def year_encodes_cleanly(year):
        a = tokenizer.encode(f" {year}")
        # make sure it tokenizes cleanly into like 1420 -> 14 and 20
        return a == [tokenizer.encode(f" {str(year)[:2]}")[0], tokenizer.encode(str(year)[2:])[0]]
    
    for century in range(11, 19):
        all_success = []
        minimum_year = century*100 + 1 # it gets confused with 00 years
        if not year_encodes_cleanly(minimum_year):
            #print(f"minimum year {minimum_year} doesn't encode cleanly into two tokens, trying a few later")
            found_minimum = False
            for offset in range(1, 3):
                minimum_year = minimum_year + offset
                if year_encodes_cleanly(minimum_year):
                    MINIMUMS[century] = minimum_year
                    found_minimum = True
                    #print(f"using minimum year {minimum_year}")
                    break
            if not found_minimum:
                #print(f"could not find a minimum year for century {century}, ignoring it")
                continue
        else:
            #print(f"using century {century}")
            MINIMUMS[century] = minimum_year
        # start a ways off of minimum year so minimum year means something
        for year in range(minimum_year+15, (century * 100) + 100):
            if year_encodes_cleanly(year):
                all_success.append(str(year))
        YEARS.extend(all_success[1:-1]) # this is to prevent stuff like 1999 (next year is a different century), that way we can just complete 19__
        
    nouns = restrict_to_most_common_size(tokenizer=tokenizer, words=NOUNS, with_space=True)
    #print("nouns using", nouns)
    nouns_perm = torch.randint(0, len(nouns), (num_patching_pairs,))
    years_perm = torch.randint(0, len(YEARS), (num_patching_pairs,))
    
    for i in range(num_patching_pairs):
        year = YEARS[years_perm[i]]
        century, decade = int(year[:2]), int(year[2:])
        correct_outputs = []
        incorrect_outputs = []
        for output_decade, tok in DECADES:
            if int(output_decade) > decade:
                correct_outputs.append(output_decade)
            else:
                incorrect_outputs.append(output_decade)
        prompt = f"The {nouns[nouns_perm[i]]} lasted from the year {year} to {century}" # century is first two tokens of year: like 1920 -> 19
        yield prompt, correct_outputs, incorrect_outputs 

        # we patch in minimum year (like for 1942, we patch in 1900, this is to make any output valid thus removing info needed to solve greater than)
        # sometimes this will be 1901 or 1902 if tokenizer doesn't encode 1900 in two tokens (for example)
        year = str(MINIMUMS[century])
        century, decade = int(year[:2]), int(year[2:])
        corrupted_correct_outputs = []
        corrupted_incorrect_outputs = []
        for output_decade, tok in DECADES:
            if int(output_decade) > decade:
                corrupted_correct_outputs.append(output_decade)
            else:
                corrupted_incorrect_outputs.append(output_decade)
        corrupted_prompt = f"The {nouns[nouns_perm[i]]} lasted from the year {year} to {century}" # century is first two tokens of year: like 1920 -> 19
        yield corrupted_prompt, corrupted_correct_outputs, corrupted_incorrect_outputs 
