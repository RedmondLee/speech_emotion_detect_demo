<template>
  <el-text tag="b" size="large" style="font-size:28px">COMP7990 - Speech Emotion Recognition DEMO</el-text><br>
  <el-text tag="b" size="large">亡者归来</el-text>
  <div style="height:20px"></div>
  <el-upload
    v-model:file-list="fileList"
    class="upload-demo"
    action="/api/uploadfile"
    multiple
    :on-preview="handlePreview"
    :on-remove="handleRemove"
    :on-success="handleSuccess"
    :before-remove="beforeRemove"
    :limit="3"
    :on-exceed="handleExceed"
  >
    <el-button type="primary">Click to upload audio file</el-button>
    <template #tip>
      <div class="el-upload__tip">
        wav/mp3 files with a size less than 500KB.
      </div>
    </template>
  </el-upload>
  <div style="height:20px"></div>
  <el-text tag="b" size="large">Your emotion is: <span :style="{'color':ecolor}">{{eresult}}</span></el-text>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import type { UploadProps, UploadUserFile } from 'element-plus'

const eresult = ref<string>('')
const ecolor = ref<string>('gray')
eresult.value = '...waiting for input...'
const fileList = ref<UploadUserFile[]>([
])

const handleRemove: UploadProps['onRemove'] = (file, uploadFiles) => {
  console.log(file, uploadFiles)
}

const handlePreview: UploadProps['onPreview'] = (uploadFile) => {
  console.log(uploadFile)
}

const handleExceed: UploadProps['onExceed'] = (files, uploadFiles) => {
  ElMessage.warning(
    `The limit is 3, you selected ${files.length} files this time, add up to ${
      files.length + uploadFiles.length
    } totally`
  )
}

const beforeRemove: UploadProps['beforeRemove'] = (uploadFile, uploadFiles) => {
  return ElMessageBox.confirm(
    `Remove file named ${uploadFile.name} ?`
  ).then(
    () => true,
    () => false
  )
}


const handleSuccess: UploadProps['onSuccess'] = (
  response,
  uploadFile
) => {
  console.log(response)
  eresult.value = response.result
  if (response.result == 'Angry') {
    ecolor.value = 'red'
  } else if (response.result == 'Happy') {
    ecolor.value = 'green'
  } else if (response.result == 'Sad') {
    ecolor.value = 'blue'
  } else if (response.result == 'Neutral') {
    ecolor.value = 'gray'
  } else if (response.result == 'Surprise') {
    ecolor.value = 'yellow'
  } else {
    ecolor.value = 'gray'
  }
}
</script>


<style scoped>
</style>
