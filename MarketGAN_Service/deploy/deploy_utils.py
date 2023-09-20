# Deploy the model to the server for web service
# import onnx
# def to_onnx(model, x, exp_name):
#     # Export the model to ONNX format
#     torch.onnx.export(model,               # model being run
#                       x,                         # model input (or a tuple for multiple inputs)
#                       f"MarketGAN_{exp_name}.onnx",   # where to save the model (can be a file or file-like object)
#                       export_params=True,        # store the trained parameter weights inside the model file
#                       opset_version=10,          # the ONNX version to export the model to
#                       do_constant_folding=True,  # whether to execute constant folding for optimization
#                       input_names = ['input'],   # the model's input names
#                       output_names = ['output'], # the model's output names
#                       dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                     'output' : {0 : 'batch_size'}})