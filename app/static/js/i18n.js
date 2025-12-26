// 国际化语言管理
const i18n = {
    currentLang: localStorage.getItem('language') || 'en',
    
    translations: {
        zh: {
            // 通用
            'lang': '语言',
            'switchLang': '切换语言',
            'back': '返回',
            'save': '保存',
            'cancel': '取消',
            'delete': '删除',
            'download': '下载',
            'loading': '加载中...',
            'error': '错误',
            'success': '成功',
            'confirm': '确认',
            
            // settings.html
            'settings': '设置',
            'settingsTitle': '设置',
            'settingsDesc': '管理您的账户和偏好设置',
            'backToChat': '返回聊天',
            'accountInfo': '账户信息',
            'username': '用户名',
            'newPassword': '新密码',
            'confirmPassword': '确认新密码',
            'newPasswordPlaceholder': '输入新密码',
            'confirmPasswordPlaceholder': '再次输入新密码',
            'saveChanges': '保存更改',
            'settingsSaved': '设置已保存',
            'loadUserInfoFailed': '加载用户信息失败',
            'passwordMismatch': '两次输入的密码不一致',
            'saveFailed': '保存失败，请稍后重试',
            
            // index.html
            'appName': '济世',
            'appSubtitle': '您的健康管理智能助手',
            'smartAnalysis': '智能分析',
            'smartAnalysisDesc': '运用先进的AI技术，为您提供专业的体检报告解读和健康建议。',
            'personalizedAdvice': '个性化建议',
            'personalizedAdviceDesc': '根据您的体检数据，提供量身定制的健康管理方案。',
            'realtimeConsultation': '实时咨询',
            'realtimeConsultationDesc': '随时随地获取专业的健康咨询服务。',
            'getStarted': '开始使用',
            
            // login.html
            'login': '登录',
            'register': '注册',
            'password': '密码',
            'registerSuccess': '注册成功，请登录',
            'loginFailed': '登录失败',
            'registerFailed': '注册失败',
            'passwordMismatchRegister': '两次输入的密码不一致',
            
            // profile.html
            'profile': '个人中心',
            'user': '用户',
            'registerTime': '注册时间',
            'lastLogin': '最后登录',
            'backToChatProfile': '返回聊天',
            'logout': '退出登录',
            'usageStats': '使用统计',
            'files': '文件',
            'totalChats': '总对话数',
            'totalMessages': '总消息数',
            'totalAssessments': '风险评估数',
            'totalReports': '导出报告数',
            'noFiles': '暂无文件',
            'noFilesDesc': '您还没有上传任何文件',
            'loadFilesFailed': '加载失败',
            'loadFilesFailedDesc': '无法加载文件列表，请稍后重试',
            'deleteFileConfirm': '确定要删除文件 "{filename}" 吗？此操作不可恢复。',
            'deleteFileFailed': '删除失败',
            'deleteFileError': '删除文件时发生错误',
            'downloadCount': '下载 {count} 次',
            
            // chat.html (主要文本)
            'chatTitle': '济世 - 智能问答',
            'newChat': '新建对话',
            'searchChats': '搜索对话',
            'sendMessage': '发送消息',
            'inputPlaceholder': '输入您的问题...',
            'clearChat': '清空对话',
            'exportChat': '导出对话',
            'deleteChat': '确定要删除这个对话吗？',
            'exporting': '导出中...',
            'exportFailed': '导出报告失败，请稍后重试',
            'deleteRiskAssessment': '确定要删除这条风险评估记录吗？',
            'deepThink': '深度思考',
            'ragNotActivated': 'RAG未启动',
            'ragActivated': 'RAG已启动',
            'webSearch': '联网搜索',
            'exportReport': '导出报告',
            'assessmentHistory': '评估历史',
            'chatMessages': '聊天消息',
            'personalCenter': '个人中心',
            'settings': '设置',
            'logout': '注销',
            'aiGeneratedNotice': '内容由 AI 生成，请仔细甄别',
            'welcomeMessage': '您好！我是济世，您的健康管理智能助手。请问有什么可以帮您？',
            'deepThinkActivated': '已深度思考',
            'aiThinking': 'AI正在思考中...',
            'collapseSidebar': '收起侧边栏',
            'expandSidebar': '展开侧边栏',
            'voiceInput': '语音输入',
            'stopVoiceInput': '点击停止语音输入',
            'menu': '菜单',
            'defaultGroup': '默认分组',
            'userInitial': '用',
            'riskAssessmentHistory': '风险评估历史记录',
            'riskAssessment': '风险评估',
            'allModels': '所有模型',
            'nurseAdvice': '护士建议',
            'featureImportanceAnalysis': '特征重要性分析',
            'paginationInfo': '共 {total} 条记录，第 {page} 页，共 {pages} 页',
            'pageInfo': '{page} / {pages}',
            'predictionResult': '预测结果',
            'formData': '表单数据',
            'predictionConfidence': '预测置信度',
            'confidenceNote': '置信度越高，预测结果越可靠',
            'deleteRecord': '删除记录',
            'predictionProbability': '预测概率',
            'historyAssessmentResult': '历史评估结果',
            'assessmentTime': '评估时间',
            'refresh': '刷新',
            'previousPage': '上一页',
            'nextPage': '下一页',
            'newConversation': '新对话',
            'startAssessment': '开始评估',
            'endAssessment': '结束评估',
            'pleaseSelect': '请选择',
            'pleaseEnter': '请输入',
            'pleaseFill': '请填写',
            'numerical': '数值',
            'updateAssessment': '更新评估',
            'submitAssessment': '提交评估',
            'loadFormFailed': '加载评估表单失败，请稍后重试',
            'updating': '更新中...',
            'assessing': '评估中...',
            'updateResult': '更新结果',
            'assessmentResult': '评估结果',
            'resultUpdatedToDatabase': '评估结果已更新到数据库',
            'resultSavedToDatabase': '评估结果已保存到数据库',
            'updateMode': '修改模式',
            'historyAssessment': '历史评估',
            'enlargeImage': '放大图片',
            'loadingHistory': '正在加载历史记录...',
            'loadingConversation': '正在加载对话...',
            'loadFailedRetry': '加载失败，请刷新页面重试',
            'loadFailed': '加载失败',
            'loadConversationFailed': '无法加载对话内容，请刷新页面重试。',
            'copy': '复制',
            'like': '赞同',
            'dislike': '反对',
            'feedbackUpdateFailed': '更新反馈失败，请稍后重试',
            'aiGeneratedImage': 'AI生成图片',
            'stage': '阶段',
            'updateCompleted': '更新完成',
            'assessmentCompleted': '评估完成',
            'updateAction': '更新',
            'saveAction': '保存',
            'submitAssessmentFailed': '提交评估失败，请稍后重试',
            'pleaseSelectChat': '请先选择一个对话',
            'speechRecognitionError': '语音识别出错: ',
            'speechNotSupported': '当前浏览器不支持语音识别',
            'microphonePermissionDenied': '麦克风权限被拒绝。请在浏览器设置中允许麦克风权限，然后刷新页面重试。',
            'microphonePermissionRequired': '需要麦克风权限才能使用语音输入功能。',
            'microphoneNotFound': '未检测到麦克风设备。请确保已连接麦克风设备，然后重试。',
            'requestPermissionAgain': '重新申请权限',
            'checkingPermission': '正在检查权限...',
            'speechRecognitionNetworkError': '语音识别网络错误。请检查网络连接，确保可以访问语音识别服务，然后重试。',
            'deleteFailed': '删除失败：',
            'copied': '已复制',
            'genericError': '抱歉，发生了一些错误，请稍后重试。',
            'addAnyContent': '添加任意内容',
            'dragDropFileHint': '将任意文件拖放到此处,以将其添加到对话中',
            'expandToolCallDetails': '展开工具调用详情',
            'toolCallResult': '工具调用结果',
            'expandReferences': '展开相关资料',
            'noModelDetected': '未检测到模型',
            'default': '默认',
            'loadFailedUseDefault': '加载失败，使用默认模型',
            'imageLoadFailed': '图片加载失败',
            'image': '图片',
            'callTool': '调用工具',
            'inputParameters': '传入参数',
            'returnResult': '返回结果',
            'truncated': '已截断'
        },
        
        en: {
            // Common
            'lang': 'Language',
            'switchLang': 'Switch Language',
            'back': 'Back',
            'save': 'Save',
            'cancel': 'Cancel',
            'delete': 'Delete',
            'download': 'Download',
            'loading': 'Loading...',
            'error': 'Error',
            'success': 'Success',
            'confirm': 'Confirm',
            
            // settings.html
            'settings': 'Settings',
            'settingsTitle': 'Settings',
            'settingsDesc': 'Manage your account and preferences',
            'backToChat': 'Back to Chat',
            'accountInfo': 'Account Information',
            'username': 'Username',
            'newPassword': 'New Password',
            'confirmPassword': 'Confirm New Password',
            'newPasswordPlaceholder': 'Enter new password',
            'confirmPasswordPlaceholder': 'Enter new password again',
            'saveChanges': 'Save Changes',
            'settingsSaved': 'Settings saved',
            'loadUserInfoFailed': 'Failed to load user information',
            'passwordMismatch': 'Passwords do not match',
            'saveFailed': 'Save failed, please try again later',
            
            // index.html
            'appName': 'JiShi',
            'appSubtitle': 'Your Intelligent Health Management Assistant',
            'smartAnalysis': 'Smart Analysis',
            'smartAnalysisDesc': 'Using advanced AI technology to provide professional health examination report interpretation and health advice.',
            'personalizedAdvice': 'Personalized Advice',
            'personalizedAdviceDesc': 'Provide tailored health management solutions based on your examination data.',
            'realtimeConsultation': 'Real-time Consultation',
            'realtimeConsultationDesc': 'Get professional health consultation services anytime, anywhere.',
            'getStarted': 'Get Started',
            
            // login.html
            'login': 'Login',
            'register': 'Register',
            'password': 'Password',
            'registerSuccess': 'Registration successful, please login',
            'loginFailed': 'Login failed',
            'registerFailed': 'Registration failed',
            'passwordMismatchRegister': 'Passwords do not match',
            
            // profile.html
            'profile': 'Profile',
            'user': 'User',
            'registerTime': 'Registration Time',
            'lastLogin': 'Last Login',
            'backToChatProfile': 'Back to Chat',
            'logout': 'Logout',
            'usageStats': 'Usage Statistics',
            'files': 'Files',
            'totalChats': 'Total Chats',
            'totalMessages': 'Total Messages',
            'totalAssessments': 'Total Risk Assessments',
            'totalReports': 'Total Exported Reports',
            'noFiles': 'No Files',
            'noFilesDesc': 'You haven\'t uploaded any files yet',
            'loadFilesFailed': 'Load Failed',
            'loadFilesFailedDesc': 'Unable to load file list, please try again later',
            'deleteFileConfirm': 'Are you sure you want to delete file "{filename}"? This action cannot be undone.',
            'deleteFileFailed': 'Delete Failed',
            'deleteFileError': 'An error occurred while deleting the file',
            'downloadCount': 'Downloaded {count} times',
            
            // chat.html (主要文本)
            'chatTitle': 'JiShi - Intelligent Q&A',
            'newChat': 'New Chat',
            'searchChats': 'Search Chats',
            'sendMessage': 'Send Message',
            'inputPlaceholder': 'Enter your question...',
            'clearChat': 'Clear Chat',
            'exportChat': 'Export Chat',
            'deleteChat': 'Are you sure you want to delete this chat?',
            'exporting': 'Exporting...',
            'exportFailed': 'Failed to export report, please try again later',
            'deleteRiskAssessment': 'Are you sure you want to delete this risk assessment record?',
            'deepThink': 'Deep Thinking',
            'ragNotActivated': 'RAG Not Activated',
            'ragActivated': 'RAG Activated',
            'webSearch': 'Web Search',
            'exportReport': 'Export Report',
            'assessmentHistory': 'Assessment History',
            'chatMessages': 'Chat Messages',
            'personalCenter': 'Personal Center',
            'settings': 'Settings',
            'logout': 'Logout',
            'aiGeneratedNotice': 'Content generated by AI, please carefully discern',
            'welcomeMessage': 'Hello! I am Jishi, your intelligent health management assistant. How can I help you?',
            'deepThinkActivated': 'Deep Thinking Activated',
            'aiThinking': 'AI is thinking...',
            'collapseSidebar': 'Collapse Sidebar',
            'expandSidebar': 'Expand Sidebar',
            'voiceInput': 'Voice Input',
            'stopVoiceInput': 'Click to stop voice input',
            'menu': 'Menu',
            'defaultGroup': 'Default Group',
            'userInitial': 'U',
            'riskAssessmentHistory': 'Risk Assessment History',
            'riskAssessment': ' Risk Assessment',
            'allModels': 'All Models',
            'nurseAdvice': 'Nurse Advice',
            'featureImportanceAnalysis': 'Feature Importance Analysis',
            'paginationInfo': 'Total {total} records, Page {page} of {pages}',
            'pageInfo': '{page} / {pages}',
            'predictionResult': 'Prediction Result',
            'formData': 'Form Data',
            'predictionConfidence': 'Prediction Confidence',
            'confidenceNote': 'Higher confidence means more reliable prediction',
            'deleteRecord': 'Delete Record',
            'predictionProbability': 'Prediction Probability',
            'historyAssessmentResult': 'History Assessment Result',
            'assessmentTime': 'Assessment Time',
            'refresh': 'Refresh',
            'previousPage': 'Previous Page',
            'nextPage': 'Next Page',
            'newConversation': 'New Conversation',
            'startAssessment': 'Start Assessment',
            'endAssessment': 'End Assessment',
            'pleaseSelect': 'Please select',
            'pleaseEnter': 'Please enter',
            'pleaseFill': 'Please fill',
            'numerical': 'Numerical',
            'updateAssessment': 'Update Assessment',
            'submitAssessment': 'Submit Assessment',
            'loadFormFailed': 'Failed to load assessment form, please try again later',
            'updating': 'Updating...',
            'assessing': 'Assessing...',
            'updateResult': 'Update Result',
            'assessmentResult': 'Assessment Result',
            'resultUpdatedToDatabase': 'Assessment result has been updated to database',
            'resultSavedToDatabase': 'Assessment result has been saved to database',
            'updateMode': 'Update Mode',
            'historyAssessment': 'History Assessment',
            'enlargeImage': 'Enlarge Image',
            'loadingHistory': 'Loading history...',
            'loadingConversation': 'Loading conversation...',
            'loadFailedRetry': 'Load failed, please refresh the page and try again',
            'loadFailed': 'Load Failed',
            'loadConversationFailed': 'Unable to load conversation, please refresh and try again.',
            'copy': 'Copy',
            'like': 'Like',
            'dislike': 'Dislike',
            'feedbackUpdateFailed': 'Failed to update feedback, please try again later',
            'aiGeneratedImage': 'AI Generated Image',
            'stage': 'Stage ',
            'updateCompleted': 'Update Completed',
            'assessmentCompleted': 'Assessment Completed',
            'updateAction': 'updated',
            'saveAction': 'saved',
            'submitAssessmentFailed': 'Failed to submit assessment, please try again later',
            'pleaseSelectChat': 'Please select a chat first',
            'speechRecognitionError': 'Speech recognition error: ',
            'speechNotSupported': 'Your browser does not support speech recognition',
            'microphonePermissionDenied': 'Microphone permission denied. Please allow microphone access in your browser settings and refresh the page.',
            'microphonePermissionRequired': 'Microphone permission is required to use voice input.',
            'microphoneNotFound': 'Microphone device not found. Please ensure a microphone is connected and try again.',
            'requestPermissionAgain': 'Request Permission Again',
            'checkingPermission': 'Checking permission...',
            'speechRecognitionNetworkError': 'Speech recognition network error. Please check your network connection and ensure you can access the speech recognition service, then try again.',
            'deleteFailed': 'Delete failed: ',
            'copied': 'Copied',
            'genericError': 'Sorry, something went wrong. Please try again later.',
            'addAnyContent': 'Add any content',
            'dragDropFileHint': 'Drag and drop any file here to add it to the conversation',
            'expandToolCallDetails': 'Expand Tool Call Details',
            'toolCallResult': 'Tool Call Result',
            'expandReferences': 'Expand References',
            'noModelDetected': 'No model detected',
            'default': 'Default',
            'loadFailedUseDefault': 'Load failed, using default model',
            'imageLoadFailed': 'Image load failed',
            'image': 'Image',
            'callTool': 'Call Tool',
            'inputParameters': 'Input Parameters',
            'returnResult': 'Return Result',
            'truncated': 'Truncated'
        }
    },
    
    // 获取翻译文本
    t(key, params = {}) {
        const translation = this.translations[this.currentLang]?.[key] || key;
        if (Object.keys(params).length > 0) {
            return translation.replace(/\{(\w+)\}/g, (match, paramKey) => {
                return params[paramKey] || match;
            });
        }
        return translation;
    },
    
    // 切换语言
    switchLanguage(lang) {
        if (this.translations[lang]) {
            this.currentLang = lang;
            localStorage.setItem('language', lang);
            this.updatePage();
        }
    },
    
    // 更新页面文本
    updatePage() {
        // 更新所有带有data-i18n属性的元素
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            
            // 如果是title属性
            if (element.hasAttribute('data-i18n-title')) {
                element.title = translation;
            }
            // 如果是其他元素，更新文本内容
            else {
                element.textContent = translation;
            }
        });
        
        // 更新所有带有data-i18n-placeholder属性的input和textarea
        document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
            const key = element.getAttribute('data-i18n-placeholder');
            const translation = this.t(key);
            element.placeholder = translation;
        });

        // 更新所有带有data-i18n-title属性的元素（即使没有data-i18n）
        document.querySelectorAll('[data-i18n-title]').forEach(element => {
            const key = element.getAttribute('data-i18n-title');
            const translation = this.t(key);
            element.title = translation;
            element.setAttribute('data-bs-original-title', translation);
        });

        // 更新所有带有data-i18n-alt属性的元素
        document.querySelectorAll('[data-i18n-alt]').forEach(element => {
            const key = element.getAttribute('data-i18n-alt');
            const translation = this.t(key);
            element.alt = translation;
        });
        
        // 更新所有带有data-i18n-button属性的按钮（动态创建的按钮）
        document.querySelectorAll('[data-i18n-button]').forEach(button => {
            const key = button.getAttribute('data-i18n-button');
            const translation = this.t(key);
            // 保留SVG图标，只更新文本部分
            const svg = button.querySelector('svg');
            if (svg) {
                // 保留完整的SVG结构，只替换文本内容
                const svgHTML = svg.outerHTML;
                button.innerHTML = svgHTML + '\n                    ' + translation;
            } else {
                button.textContent = translation;
            }
        });
        
        // 更新html lang属性
        document.documentElement.lang = this.currentLang;
        
        // 更新title
        const titleKey = document.querySelector('title')?.getAttribute('data-i18n');
        if (titleKey) {
            document.querySelector('title').textContent = this.t(titleKey);
        }
        
        // 更新拖放覆盖层的文字
        const dragTitle = document.getElementById('drag-title');
        if (dragTitle) {
            dragTitle.textContent = this.t('addAnyContent');
        }
        const dragHint = document.getElementById('drag-hint');
        if (dragHint) {
            dragHint.textContent = this.t('dragDropFileHint');
        }
    },
    
    // 初始化
    init() {
        this.updatePage();
    }
};

// 页面加载完成后初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => i18n.init());
} else {
    i18n.init();
}

