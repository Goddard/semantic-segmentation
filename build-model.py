import subprocess

def run_command(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    result = out.split(b'\n')
    for lin in result:
        if not lin.startswith(b'#'):
            print(lin)

run_command("$HOME/Libraries/TensorFlow/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=./runs/normal_pb/model.pb")

run_command('python -m tensorflow.python.tools.freeze_graph --input_graph=./runs/normal_pb/model.pb --input_binary=true --input_checkpoint=./runs/normal/model.ckpt --output_graph=./runs/freeze/model.pb --output_node_names=my_logits')

run_command('python -m tensorflow.python.tools.optimize_for_inference --input=./runs/freeze/model.pb --output=./runs/optimized/model.pb --frozen_graph=True --input_names=image_input --output_names=my_logits')

run_command("$HOME/Libraries/TensorFlow/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=$HOME/Projects/udacity/CarND-Semantic-Segmentation/runs/freeze/model.pb --out_graph=$HOME/Projects/udacity/CarND-Semantic-Segmentation/runs/eight_bit/model.pb --inputs=image_input --outputs=my_logits --transforms=' add_default_attributes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms fuse_resize_and_conv quantize_weights quantize_nodes strip_unused_nodes sort_by_execution_order'")

import re
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2

def convert_graph_to_dot(input_graph, output_dot, is_input_graph_binary):
    graph = graph_pb2.GraphDef()
    with open(input_graph, "rb") as fh:
        if is_input_graph_binary:
            graph.ParseFromString(fh.read())
        else:
            text_format.Merge(fh.read(), graph)
    with open(output_dot, "wt") as fh:
        print("digraph graphname {", file=fh)
        for node in graph.node:
            output_name = node.name
            print("  \"" + output_name + "\" [label=\"" + node.op + "\"];", file=fh)
            for input_full_name in node.input:
                parts = input_full_name.split(":")
                input_name = re.sub(r"^\^", "", parts[0])
                print("  \"" + input_name + "\" -> \"" + output_name + "\";", file=fh)
        print("}", file=fh)
        print("Created dot file '%s' for graph '%s'." % (output_dot, input_graph))

normal_pb_input_graph = './runs/normal_pb/model.pb'
normal_pb_output_dot = './runs/normal_pb/graph.dot'
convert_graph_to_dot(input_graph=normal_pb_input_graph, output_dot=normal_pb_output_dot, is_input_graph_binary=True)

freeze_input_graph = './runs/freeze/model.pb'
freeze_output_dot = './runs/freeze/graph.dot'
convert_graph_to_dot(input_graph=freeze_input_graph, output_dot=freeze_output_dot, is_input_graph_binary=True)

optimized_input_graph = './runs/optimized/model.pb'
optimized_output_dot = './runs/optimized/graph.dot'
convert_graph_to_dot(input_graph=optimized_input_graph, output_dot=optimized_output_dot, is_input_graph_binary=True)

eight_bit_input_graph = './runs/eight_bit/model.pb'
eight_bit_output_dot = './runs/eight_bit/graph.dot'
convert_graph_to_dot(input_graph=eight_bit_input_graph, output_dot=eight_bit_output_dot, is_input_graph_binary=True)

# normal_pb_dot_to_png = "dot -O -T png " + normal_pb_output_dot
# print(normal_pb_dot_to_png)
# run_command(normal_pb_dot_to_png) # + " -o " + normal_pb_output_dot + ".png > /tmp/a.out"
# print("normal pb graph png created")

freeze_dot_to_png = "dot -O -T png " + normal_pb_output_dot
print(freeze_dot_to_png)
run_command(freeze_dot_to_png) # + " -o " + freeze_output_dot + ".png > /tmp/a.out"
print("freeze graph png created")

optimized_dot_to_png = "dot -O -T png " + normal_pb_output_dot
print(optimized_dot_to_png)
run_command(optimized_dot_to_png) # + " -o " + optimized_output_dot + ".png > /tmp/a.out"
print("optimized graph png created")

eight_bit_dot_to_png = "dot -O -T png " + normal_pb_output_dot
print(eight_bit_dot_to_png)
run_command(eight_bit_dot_to_png) # + " -o " + eight_bit_output_dot + ".png > /tmp/a.out"
print("eight bit graph png created")