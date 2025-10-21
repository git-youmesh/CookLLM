import os
import torch
from datasets import load_dataset, load_metric
import pandas as pd
import ast
import re
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################

# Load the entire model on the GPU 0
device_map = {"": 0}
from peft import PeftConfig
config = PeftConfig.from_pretrained("CodeTriad/gemma_fintune_15000_2")
token =''


quantization_config = BitsAndBytesConfig(load_in_16bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it",token = token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", quantization_config=quantization_config,token = token)
model = PeftModel.from_pretrained(model, "CodeTriad/gemma_fintune_15000_2")
model.eval()
def create_csv(recipe_ids, recipes, source_ingredient_list, expected_substitute_list, suggested_substitutes_list, is_hit_list, destination):
    data = {
        "Recipe Id": recipe_ids,
        "Recipe": recipes,
        "Source Ingredient": source_ingredient_list,
        "Expected Substitute": expected_substitute_list,
        "Suggested Substitutes": suggested_substitutes_list,
        "Check": is_hit_list
    }
    df = pd.DataFrame(data)

    df.to_csv(destination, index=False)
    csv_file_path = '/content/drive/MyDrive/Colab Notebooks/FYP/test_comments_subs_with_titles.csv'

df = pd.read_csv(csv_file_path, header=None, skiprows=1)

recipe_ids = []
recipes = []
source_ingredient_list = []
expected_substitute_list = []
suggested_substitutes_list = []
isHit = []
recipe_count = 7350
num_of_hits = 399

for index, row in df.iterrows():
      ingredient_to_replace = row[2].split(",")[0].strip()  # Extract the 1st item in the 3rd column
      ingredient_name = re.sub(r'[()]', '', ingredient_to_replace).strip()  # Remove parentheses and extra spaces
      ingredient_name = ingredient_name.strip("'")  # Remove single quotes
      ingredient_name = ingredient_name.replace("_", " ")  # replace "_" with space
      substitutes = []

      recipe_title = row[3]
      expected_substitute_name = row[2].split(",")[1].strip()
      expected_substitute = re.sub(r'[()]', '',
                                    expected_substitute_name).strip()  # Remove parentheses and extra spaces
      expected_substitute = expected_substitute.strip("'")  # Remove single quotes
      expected_substitute = expected_substitute.replace("_", " ")  # replace "_" with space

      substitute_prompt = f"""<start_of_turn>user 
As a master chef, your culinary prowess knows no bounds. Your ability to flawlessly cook any dish is unparalleled.Even when faced with a missing ingredient, you effortlessly identify the perfect
substitute.
Follow the instructions below and suggest the best substitute for the given ingredient.
Instructions:
- Do not provide the same ingredient as above as the substitutes.
- Give only one ingredient.
- Avoid giving explanations.
- Only provide the name of the ingredient.
- Give the output as a numbered point.

Dish: {recipe_title}
Ingredient: {ingredient_name}
<end_of_turn>
<start_of_turn>model"""

      input_ids = tokenizer(substitute_prompt, return_tensors="pt").to("cuda")

      outputs = model.generate(**input_ids, max_new_tokens = 10)
      result = tokenizer.decode(outputs[0])

      pattern = r"\*\*Substitute:\*\*\s*(.*)"

      match = re.search(pattern, result)

      if match:
        matches = match.group(1).lower()
        matches = matches.replace("-"," ")


      try:
        recipe_ids.append(row[0])
        recipes.append(row[1])
        suggested_substitutes_list.append(matches)
        expected_substitute_list.append(expected_substitute)
        source_ingredient_list.append(ingredient_name)
        if (expected_substitute == matches):
          num_of_hits += 1
          isHit.append("TRUE")
        else:
          isHit.append("FALSE")
        recipe_count += 1
      except:
        continue
      print(matches,": ",expected_substitute, num_of_hits, recipe_count)

destination = '/content/drive/MyDrive/Colab Notebooks/FYP/outputs/gemma_finetuene_all.csv'
create_csv(recipe_ids, recipes, source_ingredient_list, expected_substitute_list, suggested_substitutes_list, isHit, destination)