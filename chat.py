import numpy as np
import re
from test_model import encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length

class ChatBot:
  
  negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")

  exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
  
  def start_chat(self):
    user_response = input("Talk to me...\n")
    
    if user_response in self.negative_responses:
      print("Ok, have a great day!")
      return
    
    self.chat(user_response.lower().replace("'", ""))
  
  def chat(self, reply):
    counter = 0
    while not self.make_exit(reply):
      if counter == 0:
        counter += 1
      else:
        with open("conversation.txt", "a+") as file_object:
            file_object.seek(0)
            file_object.write("\n")
            file_object.write(reply)    
            
      reply = input(self.generate_response(reply.lower().replace("'", "")))       
        
    
  def string_to_matrix(self, user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros(
      (1, max_encoder_seq_length, num_encoder_tokens),
      dtype='float32')
    for timestep, token in enumerate(tokens):
      if token in input_features_dict:
        user_input_matrix[0, timestep, input_features_dict[token]] = 1.
    return user_input_matrix
  
  def generate_response(self, user_input):
    input_matrix = self.string_to_matrix(user_input)
    states_value = encoder_model.predict(input_matrix)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['<START>']] = 1.
    
    chatbot_response = ''

    stop_condition = False
    while not stop_condition:
      output_tokens, hidden_state, cell_state = decoder_model.predict(
        [target_seq] + states_value)
      
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      sampled_token = reverse_target_features_dict[sampled_token_index]
      
      chatbot_response += " " + sampled_token

      if (sampled_token == '<END>' or len(re.findall(r"[\w']+|[^\s\w]", chatbot_response)) > max_decoder_seq_length):
        stop_condition = True
        
      target_seq = np.zeros((1, 1, num_decoder_tokens))
      target_seq[0, 0, sampled_token_index] = 1.
      
      states_value = [hidden_state, cell_state]
      
    chatbot_response = chatbot_response.replace("<START>", "").replace("<END>", "")
    chatbot_response += "\n"
    with open("conversation.txt", "a+") as file_object:
        file_object.seek(0)
        file_object.write("\n")
        file_object.write(chatbot_response.strip())
    
      
    return chatbot_response
  
  def make_exit(self, reply):
    for exit_command in self.exit_commands:
      if exit_command in reply:
        print("Ok, have a great day!")
        return True
      
    return False
  
malharChatter = ChatBot()
malharChatter.start_chat()


