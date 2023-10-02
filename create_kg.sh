set +xe
conda activate askBert
echo "Input file: $1"
echo "Model: $2"
cd KG-extraction
python kg-extraction.py --input_text $1
parent="$(dirname "$1")"
basename=`echo $(basename "$1")|cut -f '1' -d '.'`
cp graph.dot ${parent}/${basename}.dot
cd ../flavortext-generation
python flavortext.py --input_text $1 --run_name $2
cp graph.dot ../${basename}.dot
cp graph.gml ../${basename}.gml
cd ../