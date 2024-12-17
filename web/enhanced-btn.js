import "../../scripts/app.js";
import '../../scripts/api.js'

const htmlStyleElement = document.createElement("style");
htmlStyleElement.innerHTML = `

`

document.head.appendChild(htmlStyleElement);

const htmlDivElement = document.createElement("div");
htmlDivElement.className = "comfyui-button-group"

document.querySelector("body > div.comfyui-body-top > div > div.comfyui-menu-right > div").appendChild(htmlDivElement);

const enhancedBtn = document.createElement("button");
enhancedBtn.className = "comfyui-button";
enhancedBtn.innerHTML = "检查变量名";
enhancedBtn.onclick = async () => {
  if (await checkVarName()) {
    alert("检查通过")
  }
}

htmlDivElement.appendChild(enhancedBtn);


const exportBtn = document.createElement("button");
exportBtn.className = "comfyui-button";
exportBtn.innerHTML = "导出CM配置";
exportBtn.onclick = async () => {
  if (!await checkVarName()) {
    return false
  }
  await exportPrompt()
}

htmlDivElement.appendChild(exportBtn);

async function checkVarName() {
  const p = await app.graphToPrompt()
  const setName = new Set()
  let findConfig = false
  let findOutput = false
  for (const node of p.workflow.nodes) {
    const name = node.type
    if (name === "ServiceConfigNode") {
      if (findConfig) {
        alert("找到多个ServiceConfigNode")
        return false
      } else {
        findConfig = true
      }
    }
    if (!name.startsWith("CMaster_")) {
      continue;
    }
    if (name.startsWith("CMaster_Output")) {
      findOutput = true
    }

    if (!node.widgets_values) {
      alert(`${name} 没有变量名`)
      return false
    }
    if (node.widgets_values.length === 0) {
      alert(`${name} 没有属性`)
      return false
    }
    if (node.widgets_values[0] === undefined) {
      alert(`${name} 属性名称没有配置`)
      return false
    }
    if ( typeof node.widgets_values[0] !== "string") {
      alert(`${name} 属性名称不是字符串`)
      return false
    }
    if (setName.has(node.widgets_values[0])) {
      alert(`${name} ${node.widgets_values[0]} 属性名称重复`)
      return false
    }
    setName.add(node.widgets_values[0])
  }
  if (!findConfig) {
    alert("没有找到ServiceConfigNode")
    return false
  }
  if (!findOutput) {
    alert("没有找到CMaster_Output节点")
    return false
  }
  return true
}

async function exportPrompt() {
  const p = await app.graphToPrompt()
  let nodeId = ""
  let serviceName = ""
  let serviceDescription = ""
  let serviceCode = "Code_" + Date.now()
  for (const node of p.workflow.nodes) {
    const name = node.type
    if (name === "ServiceConfigNode") {
      serviceName = node.widgets_values[0]
      serviceDescription = node.widgets_values[1]
      nodeId = node.id
      break
    }
  }
  const workflow = {...p.output}
  delete workflow[nodeId]
  const saveObj = {
    code: serviceCode,
    name: serviceName,
    description: serviceDescription,
    workflow: JSON.stringify(workflow),
    params: Object.values(workflow).filter(e => e["class_type"].startsWith("CMaster_Input")).map(e => parseInput(e)),
    outputs: Object.values(workflow).filter(e => e["class_type"].startsWith("CMaster_Output")).map(e => parseInput(e))
  }
  exportJson(serviceCode, JSON.stringify(saveObj, null, 2))
}

const ParameterType = {
  None: 0,
  String: 1,
  Number: 2,
  Boolean: 3,
  Image: 4,
  Range_Number: 5,
  Enum_String: 6,
  Float: 7,
  Range_Float: 8,
}

function parseInput(node) {
  const type = node["class_type"]
  const varName = node.inputs["var_name"]
  const newVarName = `ComfyMasterVar_${varName}`;
  const description = node.inputs["description"] || varName
  const isExport = node.inputs["export"]
  let ret= {
    key: newVarName,
    name: description,
    type: ParameterType.Image,
    isExport: isExport,
  };

  if (type === "CMaster_InputImage") {
    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.Image,
      isExport: isExport,
    }
  } else if (type === "CMaster_InputString") {
    const text = node.inputs["text"]

    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.String,
      isExport: isExport,
      stringDefaultValue: text,
    }
  } else if (type === "CMaster_InputEnumString") {
    const text = node.inputs["text"]
    const enums = node.inputs["enums"]

    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.Enum_String,
      isExport: isExport,
      stringDefaultValue: text,
      enumStringValue: enums,
    }
  } else if (type === "CMaster_InputBoolean") {
    const num = node.inputs["value"]

    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.Boolean,
      isExport: isExport,
      boolDefaultValue: num,
    }
  } else if (type === "CMaster_InputInt") {
    const num = node.inputs["number"]

    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.Number,
      isExport: isExport,
      numberDefaultValue: num,
    }
  } else if (type === "CMaster_InputRangeInt") {
    const num = node.inputs["number"]
    const min = node.inputs["min"]
    const max = node.inputs["max"]

    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.Range_Number,
      isExport: isExport,
      numberDefaultValue: num,
      minNumberValue: min,
      maxNumberValue: max,
    }
  } else if (type === "CMaster_InputFloat") {
    const num = node.inputs["number"]

    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.Float,
      isExport: isExport,
      floatDefaultValue: num,
    }
  } else if (type === "CMaster_InputRangeFloat") {
    const num = node.inputs["number"]
    const min = node.inputs["min"]
    const max = node.inputs["max"]

    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.Range_Float,
      isExport: isExport,
      floatDefaultValue: num,
      minFloatValue: min,
      maxFloatValue: max,
    }
  } else if (type === "CMaster_InputCheckpoint") {
    const text = node.inputs["ckpt_name"]
    const enums = node.inputs["checkpoints"]

    ret = {
      key: newVarName,
      name: description,
      type: ParameterType.Enum_String,
      isExport: isExport,
      stringDefaultValue: text,
      enumStringValue: enums,
    }
  }

  return ret;
}

const OutputType = {
  None: 0,
  String: 1,
  Image: 2,
  Video: 3,
  Audio: 4,
}

function parseOutput(node) {
  const type = node["class_type"]
  const varName = node.inputs["var_name"]
  const newVarName = `ComfyMasterVar_${varName}`;
  const description = node.inputs["description"]
  const isExport = node.inputs["export"]
  let ret = {
    key: newVarName,
    name: description,
    type: OutputType.None,
    isExport: isExport,
  }

  if (type === "CMaster_OutputImage") {
    ret.type = OutputType.Image;
  }
  return  ret;
}

function getFilename(defaultName) {
  if (app.ui.settings.getSettingValue('Comfy.PromptFilename', true)) {
    defaultName = prompt('保存为:', defaultName)
    if (!defaultName) return
    if (!defaultName.toLowerCase().endsWith('.json')) {
      defaultName += '.json'
    }
  }
  return defaultName
}

function exportJson(id, json) {
  const blob = new Blob([json], { type: 'application/json' })
  const file = getFilename(`export_${id}.json`)
  if (!file) return
  comfyAPI.utils.downloadBlob(file, blob)
}