vote_prompt = '''Given a question and several choices of next steps, analyze each choice in detail and compare them to decide which choice is the most promising to be the next step to solve the question. After analyzing each choice in detail and comparing them, conclude your final choice with \"Therefore, the best choice is\".
Example:

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Choice 1: There are 15 trees originally.
Choice 2: There are originally 3 cars.
Choice 3: There are 20 trees originally.
Response: Choice 1 logically follows the progression of the problem: recognizing the original amount. However, Choice 2 involves irrelevant information (3 cars) and Choice 3 contains incorrect information (20 trees). Therefore, the best choice is 1.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Choice 1: Start with 3 cars. 2 more cars arrive, making it 3 + 2 = 5.
Choice 2: There are 3 cars. If 3 more cars leave, it's 3 - 3 = 0.
Choice 3: Originally 3 cars. Double the number, 3 * 2 = 6.
Response: Choice 1 is the only one that accurately reflects the scenario: starting with the initial number of cars and adding the ones that arrive. Choices 2 and 3 involve operations that are not relevant to the situation described in the question. Therefore, the best choice is 1.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Choice 1: Leah had 32 chocolates. Her sister had 42. Total is 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
Choice 2: Leah had 32 chocolates. If she gives half to her sister, it's 32 / 2 = 16. Not relevant to the question.
Choice 3: They start with 32 and 42 chocolates. If they lose some, say 10, it's 32 - 10 = 22 and 42 - 10 = 32. Total 22 + 32 = 54.
Response: Choice 1 correctly calculates the total number of chocolates and then subtracts the amount they ate, providing the correct remaining total. Choices 2 and 3 involve calculations or scenarios that do not align with the question. Therefore, the best choice is 1.

Question: {instruction}
'''


cot_prompt = '''Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next best step to solve the problem which may involve arithmetic calculation. Please start your answer with "Let's think step by step" in the very first sentence. When the original question is answerable, please start the subquestion with "The answer is".
IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
Solved Example:

Question: Genghis Khan: founder and first Great Khan of the Mongol Empire. Are more people today related to Genghis Khan than Julius Caesar?
Step 1: Genghis Khan was the founder and first Great Khan of the Mongol Empire, which became the largest contiguous empire in history. He had numerous children, and his descendants spread widely across Asia.
Step 2: Julius Caesar was a Roman general and statesman who played a critical role in the events leading to the demise of the Roman Republic and the rise of the Roman Empire. His lineage is not as widely dispersed as Genghis Khan's.
Step 3: Studies have shown that a significant proportion of men in Asia, roughly 0.5 percent of the world's male population, carry a Y-chromosome lineage traced back to Genghis Khan. This suggests that many people today are directly descended from him.
Step 4: Julius Caesar, while a significant historical figure, does not have a documented widespread genetic legacy comparable to Genghis Khan.
Step 5: Based on historical and genetic evidence, more people today are likely related to Genghis Khan than Julius Caesar.
Step 6: The answer is true.

Question: The Police: English rock band. Could the members of The Police perform lawful arrests?
Step 1: The Police is an English rock band known for its music, not for any official law enforcement duties.
Step 2: Lawful arrests are actions typically carried out by individuals who have legal authority, such as police officers, sheriffs, or other law enforcement officials.
Step 3: The members of The Police band, being musicians, do not have the legal authority to perform arrests unless they hold some official position in law enforcement, which is not indicated.
Step 4: Therefore, based on their known roles as musicians, the members of The Police band cannot perform lawful arrests.
Step 5: The answer is false.

Question: Depression (mood): state of low mood and fatigue. Would a Monoamine Oxidase candy bar cheer up a depressed friend?
Step 1: Depression is a state of low mood and fatigue that can have various causes, including biological, psychological, and social factors.
Step 2: Monoamine Oxidase (MAO) is an enzyme that breaks down monoamines, which are neurotransmitters such as serotonin, norepinephrine, and dopamine, associated with mood regulation.
Step 3: MAO inhibitors (MAOIs) are a class of medications that inhibit the activity of Monoamine Oxidase, thereby increasing the levels of these neurotransmitters and potentially improving mood in people with depression.
Step 4: A "Monoamine Oxidase candy bar" is not a recognized treatment and is likely a fictional or hypothetical product. Consuming such a candy bar would not be a safe or effective way to manage depression.
Step 5: Effective treatment of depression typically involves a combination of therapy, medication, lifestyle changes, and support from healthcare professionals.
Step 6: Therefore, it is unlikely that a Monoamine Oxidase candy bar would cheer up a depressed friend.
Step 7: The answer is false.

Question: Grey seal: species of seal. Would a dog respond to bell before Grey seal?
Step 1: Dogs are known to be highly trainable animals and have been extensively studied in the context of classical conditioning, most famously by Ivan Pavlov. Pavlov's experiments demonstrated that dogs could be trained to associate the sound of a bell with food, eventually responding to the bell alone by salivating.
Step 2: Grey seals, while intelligent and capable of learning, are not commonly studied or trained using classical conditioning methods in the same way dogs are.
Step 3: The speed and efficiency of response to conditioning (such as responding to a bell) are generally higher in animals that have been domesticated and frequently trained by humans, such as dogs.
Step 4: Therefore, it is reasonable to conclude that a dog would likely respond to a bell more quickly and reliably than a Grey seal, given the extensive history and evidence of training in dogs.
Step 5: The answer is true.

Question: Shrimp: Decapod crustaceans. Is shrimp scampi definitely free of plastic?
Step 1: Shrimp are decapod crustaceans that are often used in various culinary dishes, including shrimp scampi.
Step 2: Shrimp scampi is a popular dish typically made with shrimp, garlic, butter, white wine, and lemon juice, served over pasta or rice.
Step 3: The presence of plastic in food can result from environmental pollution, where microplastics enter the ocean and are ingested by marine life, including shrimp.
Step 4: While the ingredients of shrimp scampi do not inherently include plastic, the possibility of microplastic contamination in seafood, including shrimp, has been documented due to pollution.
Step 5: Therefore, it cannot be definitively stated that shrimp scampi is completely free of plastic without testing for contamination.
Step 6: The answer is false.

Question: {input}
'''