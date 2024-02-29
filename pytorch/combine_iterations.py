# import torch

# # Define the list of paths to your individual .pth files
# pth_files = ['/media/tony/ASUS_PROG-FILES/Marikaiti/DIT/ptyxiakh/midifiles/piano_trans/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=1/0_iterations.pth',
# '/media/tony/ASUS_PROG-FILES/Marikaiti/DIT/ptyxiakh/midifiles/piano_trans/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=1/1_iterations.pth']

# # Initialize an empty state dictionary
# combined_state_dict = {}

# # Loop through each .pth file
# for pth_file in pth_files:
#     # Load the state dictionary from the current file
#     state_dict = torch.load(pth_file)
    
#     # Merge the parameters from the current state dictionary into the combined state dictionary
#     combined_state_dict.update(state_dict)

# # Save the combined state dictionary to a new .pth file
# torch.save(combined_state_dict, '/media/tony/ASUS_PROG-FILES/Marikaiti/DIT/ptyxiakh/midifiles/piano_trans/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/combined_note_model.pth')



import torch

full_checkpoint = torch.load('/media/tony/ASUS_PROG-FILES/Marikaiti/DIT/ptyxiakh/piano_transcription_inference_data/note_F1=0.9677_pedal_F1=0.9186.pth')
predal_model_weights = full_checkpoint['model']['pedal_model']     #extract only the weights of the note transcription


# Save the combined state dictionary to a new .pth file
torch.save(predal_model_weights, '/media/tony/ASUS_PROG-FILES/Marikaiti/DIT/ptyxiakh/midifiles/piano_trans/checkpoints/main/pedal_model.pth')