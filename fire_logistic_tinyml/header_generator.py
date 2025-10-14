import os

def tflite_to_c_header(tflite_path, header_path, array_name="model_tflite"):
    b = open(tflite_path, "rb").read()
    with open(header_path, "w") as f:
        guard = array_name.upper() + "_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"// {os.path.basename(tflite_path)} - {len(b)} bytes\n")
        f.write(f"const unsigned int {array_name}_len = {len(b)};\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")
        for i in range(0, len(b), 12):
            chunk = b[i:i+12]
            f.write("  " + ", ".join(str(x) for x in chunk) + ",\n")
        f.write("};\n\n#endif // " + guard + "\n")
    print("Wrote header:", header_path)