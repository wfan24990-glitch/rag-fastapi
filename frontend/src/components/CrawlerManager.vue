<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>Crawler Manager (is.nju.edu.cn)</span>
        <el-tag v-if="status.last_run" :type="getStatusType(status.last_run.status)">
          {{ status.last_run.status }}
        </el-tag>
      </div>
    </template>

    <div class="stats-row">
      <el-statistic title="Last Sync Date" :value="status.last_sync_date || 'Never'" />
      <el-statistic title="Total Indexed URLs" :value="status.total_seen_urls" />
    </div>

    <el-divider />

    <div class="actions-row">
      <el-button type="primary" @click="runCrawler('incremental')" :loading="loading">
        Incremental Update
      </el-button>
      <el-button type="warning" @click="runCrawler('full')" :loading="loading">
        Full Crawl (Max 50 Pages)
      </el-button>
      <el-button @click="runCrawler('incremental', true)" :loading="loading">
        Dry Run (Test)
      </el-button>
    </div>

    <div v-if="status.last_run" class="run-details">
      <h4>Last Run Details</h4>
      <p>Run ID: {{ status.last_run.run_id }}</p>
      <p>Mode: {{ status.last_run.mode }}</p>
      <p>Ingested: {{ status.last_run.ingested_count }} articles</p>
      <p>Skipped: {{ status.last_run.skipped_count }} articles</p>
      <p v-if="status.last_run.error_count > 0" class="error-text">
        Errors: {{ status.last_run.error_count }}
      </p>
      <el-collapse v-if="status.last_run.errors.length > 0">
        <el-collapse-item title="Error Logs" name="1">
          <div v-for="(err, idx) in status.last_run.errors" :key="idx" class="log-item">
            {{ err }}
          </div>
        </el-collapse-item>
      </el-collapse>
    </div>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const status = ref({
  last_sync_date: null,
  total_seen_urls: 0,
  last_run: null
})
const loading = ref(false)
let pollInterval = null

const fetchStatus = async () => {
  try {
    const res = await axios.get('/api/crawler/status')
    status.value = res.data
    
    // Stop polling if completed or failed
    if (status.value.last_run && ['completed', 'failed'].includes(status.value.last_run.status)) {
      if (pollInterval) {
        clearInterval(pollInterval)
        pollInterval = null
        loading.value = false
      }
    }
  } catch (e) {
    console.error("Failed to fetch status", e)
  }
}

const runCrawler = async (mode, dryRun = false) => {
  loading.value = true
  try {
    await axios.post('/api/crawler/run', {
      mode: mode,
      max_pages: mode === 'full' ? 50 : 10,
      dry_run: dryRun
    })
    ElMessage.success('Crawler started')
    fetchStatus()
    // Start polling
    if (!pollInterval) {
      pollInterval = setInterval(fetchStatus, 2000)
    }
  } catch (e) {
    ElMessage.error('Failed to start crawler')
    loading.value = false
  }
}

const getStatusType = (s) => {
  if (s === 'running') return 'primary'
  if (s === 'completed') return 'success'
  if (s === 'failed') return 'danger'
  return 'info'
}

onMounted(() => {
  fetchStatus()
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.stats-row {
  display: flex;
  gap: 40px;
  margin-bottom: 20px;
}
.actions-row {
  margin-bottom: 20px;
  display: flex;
  gap: 10px;
}
.run-details {
  background: #f5f7fa;
  padding: 15px;
  border-radius: 4px;
}
.error-text {
  color: #f56c6c;
}
.log-item {
  font-family: monospace;
  font-size: 12px;
  color: #666;
}
</style>
