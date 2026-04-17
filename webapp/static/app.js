function appData() {
    return {
        // State
        jobs: [],
        activeJobs: [],
        completedJobs: [],
        uploadName: '',
        uploadFile: null,
        uploading: false,
        uploadProgress: 0,
        detailJob: null,
        detailContent: '',
        editingName: false,
        editNameVal: '',
        searchOpen: false,
        searchQuery: '',
        searchResults: null,
        eventSource: null,
        theme: 'dark',

        init() {
            this.theme = localStorage.getItem('theme') || 'dark';
            this.applyTheme();
            this.loadJobs();
            this.connectSSE();
        },

        // ── Theme ─────────────────────────────────────

        toggleTheme() {
            this.theme = this.theme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('theme', this.theme);
            this.applyTheme();
        },

        applyTheme() {
            const html = document.documentElement;
            if (this.theme === 'dark') {
                html.classList.add('dark');
            } else {
                html.classList.remove('dark');
            }
        },

        // ── Data loading ──────────────────────────────

        async loadJobs() {
            const resp = await fetch('/api/jobs');
            this.jobs = await resp.json();
            this.categorizeJobs();
        },

        categorizeJobs() {
            this.activeJobs = this.jobs.filter(
                j => j.status === 'pending' || j.status === 'processing'
            );
            this.completedJobs = this.jobs.filter(
                j => j.status !== 'pending' && j.status !== 'processing'
            );
        },

        // ── SSE ───────────────────────────────────────

        connectSSE() {
            if (this.eventSource) {
                this.eventSource.close();
            }
            this.eventSource = new EventSource('/api/progress');
            this.eventSource.onmessage = (event) => {
                const active = JSON.parse(event.data);
                // Merge active job updates into our state
                for (const updated of active) {
                    const idx = this.jobs.findIndex(j => j.id === updated.id);
                    if (idx >= 0) {
                        this.jobs[idx] = { ...this.jobs[idx], ...updated };
                    }
                }
                // Check if any previously active job is now missing from active list
                const activeIds = new Set(active.map(j => j.id));
                const wasActive = this.activeJobs.some(j => !activeIds.has(j.id));
                if (wasActive) {
                    // A job finished/cancelled — reload all jobs to get final state
                    this.loadJobs();
                }
                this.categorizeJobs();
            };
            this.eventSource.onerror = () => {
                this.eventSource.close();
                this.eventSource = null;
                setTimeout(() => this.connectSSE(), 3000);
            };
        },

        // ── Upload ────────────────────────────────────

        upload() {
            if (!this.uploadFile || !this.uploadName.trim()) return;

            this.uploading = true;
            this.uploadProgress = 0;

            const formData = new FormData();
            formData.append('name', this.uploadName.trim());
            formData.append('file', this.uploadFile);

            const xhr = new XMLHttpRequest();
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    this.uploadProgress = Math.round((e.loaded / e.total) * 100);
                }
            };
            xhr.onload = () => {
                this.uploading = false;
                if (xhr.status === 201) {
                    this.uploadName = '';
                    this.uploadFile = null;
                    // Reset file input
                    const fileInput = document.querySelector('input[type="file"]');
                    if (fileInput) fileInput.value = '';
                    this.loadJobs();
                } else {
                    alert('Upload failed: ' + xhr.responseText);
                }
            };
            xhr.onerror = () => {
                this.uploading = false;
                alert('Network error while uploading.');
            };
            xhr.open('POST', '/api/jobs');
            xhr.send(formData);
        },

        // ── Job actions ───────────────────────────────

        async cancelJob(jobId) {
            await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
            this.loadJobs();
        },

        async deleteJob(jobId) {
            await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
            this.detailJob = null;
            this.detailContent = '';
            this.loadJobs();
        },

        async toggleKeepVideo(jobId, keep) {
            await fetch(`/api/jobs/${jobId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ keep_video: keep ? 1 : 0 }),
            });
            this.loadJobs();
        },

        // ── Transcription detail ──────────────────────

        async openTranscription(job) {
            this.detailJob = job;
            this.detailContent = '';
            this.editingName = false;
            if (job.status === 'completed') {
                const resp = await fetch(`/api/jobs/${job.id}/transcription`);
                if (resp.ok) {
                    const data = await resp.json();
                    this.detailContent = data.content;
                } else {
                    this.detailContent = 'Failed to load transcription.';
                }
            } else if (job.status === 'failed') {
                this.detailContent = 'Error: ' + (job.error_message || 'unknown');
            }
        },

        async openTranscriptionById(jobId) {
            const resp = await fetch(`/api/jobs/${jobId}`);
            if (resp.ok) {
                const job = await resp.json();
                this.openTranscription(job);
            }
        },

        async saveName() {
            if (!this.editNameVal.trim() || !this.detailJob) return;
            await fetch(`/api/jobs/${this.detailJob.id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: this.editNameVal.trim() }),
            });
            this.detailJob.name = this.editNameVal.trim();
            this.editingName = false;
            this.loadJobs();
        },

        // ── Search ────────────────────────────────────

        async doSearch() {
            if (!this.searchQuery.trim()) {
                this.searchResults = null;
                return;
            }
            const resp = await fetch(`/api/search?q=${encodeURIComponent(this.searchQuery)}`);
            this.searchResults = await resp.json();
        },

        // ── Shutdown ──────────────────────────────────

        async shutdown() {
            await fetch('/api/shutdown', { method: 'POST' });
        },

        // ── Helpers ───────────────────────────────────

        formatDate(iso) {
            if (!iso) return '';
            const d = new Date(iso);
            return d.toLocaleDateString('en-GB', {
                day: '2-digit', month: 'short', year: 'numeric',
                hour: '2-digit', minute: '2-digit',
            });
        },

        formatDuration(seconds) {
            if (!seconds) return '';
            const m = Math.floor(seconds / 60);
            return m + ' min';
        },
    };
}
