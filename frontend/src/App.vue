<script setup>
import RAGDebug from './components/RAGDebug.vue'
import CrawlerManager from './components/CrawlerManager.vue'
</script>

<template>
  <div class="app-container">
    <CrawlerManager style="margin-bottom: 20px;" />
    <RAGDebug />
  </div>
</template>

<style>
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background-color: #f0f2f5;
}
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}
</style>
