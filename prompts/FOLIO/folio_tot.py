vote_prompt = '''Given a question and several choices of next steps, analyze each choice in detail and compare them to decide which choice is the most promising to be the next step to solve the question. After analyzing each choice in detail and comparing them, conclude your final choice with \"Therefore, the best choice is <number>\".

Question: Some students in the class who are good at math are also good at chemistry. All students in the class who are good at chemistry enjoy conducting experiments. All students in the class that enjoy conducting experiments are good at planning. None of the students who are good at planning failed the class. James is a student in the class; he is either good at chemistry and failed the class, or bad at chemistry and passed the class. Based on the above information, is the following statement true, false, or uncertain? James is good at planning.
Choice 1: If James is bad at chemistry, he doesn't necessarily enjoy experiments or excel in planning, aligning with the premises but not directly confirming his planning skills.
Choice 2: James is good at chemistry, thus enjoys conducting experiments and is good at planning. But since he failed the class, he cannot be good at planning.
Choice 3: Suggesting James might be good at planning despite his chemistry skills is unsupported and contradicts the established logic.
Response: Choice 2 logically connects the attributes of being good at chemistry to enjoying experiments and being good at planning, but refutes the idea that James could be good at planning if he failed the class. Choice 1 correctly suggests that if James is bad at chemistry, it does not necessarily imply he is good at planning, but it doesn't conclude definitively about his planning skills. Choice 3 introduces an unsupported possibility that James might be good at planning, which contradicts the logic that only students who don't fail are good at planning. Therefore, the best choice is 2.

Question: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant. Based on the above information, is the following statement true, false, or uncertain? A Czech published a book in 1946.
Choice 1: Miroslav Venhoda was a Czech and published a book in 1946.
Choice 2: Since some musicians love music, it is likely that a Czech published a book in 1946.
Choice 3: As a choral conductor, Miroslav Venhoda focused on Renaissance and Baroque music, which does not necessarily involve publishing a book.
Response: Choice 1 directly connects the facts given about Miroslav Venhoda being Czech and his publication of a book in 1946, affirming the statement. Choice 2 introduces a generalization about musicians that does not specifically support the statement about a Czech publishing a book in 1946. Choice 3, while correctly describing his focus on certain types of music, fails to address the publication of the book in 1946, which is specifically stated. Therefore, the best choice is 1.

Question: Lana Wilson directed After Tiller, The Departure, and Miss Americana. If a film is directed by a person, the person is a filmmaker. After Tiller is a documentary. The documentary is a type of film. Lana Wilson is from Kirkland. Kirkland is a US city. If a person is from a city in a country, the person is from the country. After Tiller is nominated for the Independent Spirit Award for Best Documentary. Based on the above information, is the following statement true, false, or uncertain? Miss Americana is not directed by a filmmaker from Kirkland.
Choice 1: Because After Tiller received a nomination and Lana Wilson directed it, it does not directly imply anything about the location of the filmmaker who directed Miss Americana.
Choice 2: Miss Americana is a film directed by Lana Wilson, but not necessarily documentary-focused like After Tiller, which does not address her origins or where she is from.
Choice 3: Since Lana Wilson directed Miss Americana and is from Kirkland, the statement that Miss Americana is not directed by a filmmaker from Kirkland is false.
Response: Choice 3 clearly refutes the given statement by establishing that Lana Wilson, who is indeed from Kirkland, directed Miss Americana, making the claim false. Choice 1 and Choice 2 do not address the specific query about the filmmaker’s origin in relation to directing Miss Americana. Therefore, the best choice is 3.

Question: {instruction}
'''


cot_prompt = '''Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next best step to solve the problem which may involve arithmetic calculation. Please start your answer with \"Let's think step by step\" in the very first sentence. When the original question is answerable, please start the subquestion with \"The answer is\". IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.

Question: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant. Based on the above information, is the following statement true, false, or uncertain? A Czech published a book in 1946.
Step 1: Miroslav Venhoda is identified as a Czech choral conductor.
Step 2: It is stated that Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
Step 3: Therefore, we can conclude that a Czech (Miroslav Venhoda) published a book in 1946.
Step 4: The answer is true.

Question: Some students in the class who are good at math are also good at chemistry. All students in the class who are good at chemistry enjoy conducting experiments. All students in the class that enjoy conducting experiments are good at planning. None of the students who are good at planning failed the class. James is a student in the class; he is either good at chemistry and failed the class, or bad at chemistry and passed the class. Based on the above information, is the following statement true, false, or uncertain? James is good at planning.
Step 1: Some students are good at both math and chemistry.
Step 2: All students who are good at chemistry enjoy conducting experiments.
Step 3: All students who enjoy conducting experiments are good at planning.
Step 4: None of the students who are good at planning failed the class.
Step 5: James either failed the class being good at chemistry, or passed the class being bad at chemistry.
Step 6: If James was good at chemistry and failed, he cannot be good at planning, because those good at planning do not fail.
Step 7: If James passed the class and is bad at chemistry, he would not necessarily be good at conducting experiments or planning.
Step 8: Therefore, we cannot confirm that James is good at planning without knowing more specifics about his success in chemistry and class status.
Step 9: The answer is uncertain.

Question: Lana Wilson directed After Tiller, The Departure, and Miss Americana. If a film is directed by a person, the person is a filmmaker. After Tiller is a documentary. The documentary is a type of film. Lana Wilson is from Kirkland. Kirkland is a US city. If a person is from a city in a country, the person is from the country. After Tiller is nominated for the Independent Spirit Award for Best Documentary. Based on the above information, is the following statement true, false, or uncertain? Miss Americana is not directed by a filmmaker from Kirkland.
Step 1: Lana Wilson is identified as the director of After Tiller, The Departure, and Miss Americana.
Step 2: Lana Wilson is from Kirkland, which is a US city, indicating that Lana Wilson is from the US.
Step 3: Since Lana Wilson directed Miss Americana and she is from Kirkland, it follows that Miss Americana was directed by a filmmaker from Kirkland.
Step 4: The statement that Miss Americana is not directed by a filmmaker from Kirkland is false. 
Step 5: The answer is false.

Question: {input}
'''
