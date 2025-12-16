<template>
  <div class="common-layout">
    <el-container>
      <el-header>
        <h1>RAG Debug Console</h1>
      </el-header>
      <el-main>
        <el-card class="box-card">
          <template #header>
            <div class="card-header">
              <span>Query</span>
            </div>
          </template>
          <el-input
            v-model="query"
            placeholder="Please input query"
            @keyup.enter="handleQuery"
            clearable
          >
            <template #append>
              <el-button @click="handleQuery" :loading="loading">Search</el-button>
            </template>
          </el-input>
        </el-card>

        <el-row :gutter="20" style="margin-top: 20px;" v-if="result">
          <el-col :span="16">
            <el-card class="box-card">
              <template #header>
                <div class="card-header">
                  <span>Answer</span>
                </div>
              </template>
              <div style="white-space: pre-wrap;">{{ result.answer }}</div>
            </el-card>

            <el-card class="box-card" style="margin-top: 20px;">
              <template #header>
                <div class="card-header">
                  <span>Retrieval Process</span>
                </div>
              </template>
              
              <el-tabs v-model="activeTab">
                <el-tab-pane label="Reranked (Final Context)" name="reranked">
                  <el-table :data="result.debug_info.retrieval.reranked_candidates" style="width: 100%">
                    <el-table-column prop="score" label="Score" width="100">
                      <template #default="scope">
                        {{ scope.row.score.toFixed(4) }}
                      </template>
                    </el-table-column>
                    <el-table-column prop="source" label="Source" width="150" />
                    <el-table-column prop="text" label="Text" />
                  </el-table>
                </el-tab-pane>
                <el-tab-pane label="Initial Candidates" name="initial">
                  <el-table :data="result.debug_info.retrieval.initial_candidates" style="width: 100%">
                    <el-table-column prop="score" label="Score" width="100">
                      <template #default="scope">
                        {{ scope.row.score.toFixed(4) }}
                      </template>
                    </el-table-column>
                    <el-table-column prop="source" label="Source" width="150" />
                    <el-table-column prop="text" label="Text" />
                  </el-table>
                </el-tab-pane>
              </el-tabs>
            </el-card>
          </el-col>
          
          <el-col :span="8">
            <el-card class="box-card">
              <template #header>
                <div class="card-header">
                  <span>Timings</span>
                </div>
              </template>
              <el-timeline>
                <el-timeline-item timestamp="Embedding" placement="top">
                  {{ (result.debug_info.timings.embedding * 1000).toFixed(2) }} ms
                </el-timeline-item>
                <el-timeline-item timestamp="Vector Search" placement="top">
                  {{ (result.debug_info.timings.search * 1000).toFixed(2) }} ms
                </el-timeline-item>
                <el-timeline-item timestamp="Rerank" placement="top">
                  {{ (result.debug_info.timings.rerank * 1000).toFixed(2) }} ms
                </el-timeline-item>
                <el-timeline-item timestamp="Generation" placement="top">
                  {{ (result.debug_info.timings.generation * 1000).toFixed(2) }} ms
                </el-timeline-item>
                <el-timeline-item timestamp="Total" placement="top" type="primary">
                  {{ (result.debug_info.timings.total * 1000).toFixed(2) }} ms
                </el-timeline-item>
              </el-timeline>
            </el-card>
          </el-col>
        </el-row>
      </el-main>
    </el-container>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const query = ref('')
const loading = ref(false)
const result = ref(null)
const activeTab = ref('reranked')

const handleQuery = async () => {
  if (!query.value) return
  
  loading.value = true
  result.value = null
  
  try {
    const response = await axios.post('/api/query', {
      query: query.value,
      top_k: 20 // Fetch more candidates initially to see reranking effect
    })
    result.value = response.data
  } catch (error) {
    console.error(error)
    ElMessage.error('Query failed: ' + (error.response?.data?.detail || error.message))
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.common-layout {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
