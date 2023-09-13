
# import grpc
# from grpc4bmi.bmi_grpc_client import BmiClient

# mymodel = BmiClient(grpc.insecure_channel("localhost:5000"))
# print(mymodel.get_component_name())

#不用跑runbmiserver，BmiClientSubProcess函数包含run-bmi-server
from grpc4bmi.bmi_client_subproc import BmiClientSubProcess
mymodel = BmiClientSubProcess(path = "/home/wangjingyi/code/hydro-model-xaj",module_name = "xaj.xaj_bmi.xajBmi")
print(mymodel.get_component_name())