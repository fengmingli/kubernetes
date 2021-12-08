
## kube-scheduler的Extender解析
- Extender是kube-scheduler抽象的调度扩展程序接口，kube-scheduler在调度流程的适当地方调用Extender协助kube-scheduler完成调度；
- Extender可以实现过滤、评分、抢占和绑定，主要用来实现Kubernetes无法管理的资源的调度；
- HTTPExtender是Extender的一种实现，用于将Extender的接口请求转为HTTP请求发送给调度扩展程序，配置调度扩展程序通过kube-scheduler的配置文件实现；
