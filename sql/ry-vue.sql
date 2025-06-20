/*
 Navicat Premium Dump SQL

 Source Server         : revi
 Source Server Type    : MySQL
 Source Server Version : 80040 (8.0.40)
 Source Host           : localhost:3306
 Source Schema         : ry-vue

 Target Server Type    : MySQL
 Target Server Version : 80040 (8.0.40)
 File Encoding         : 65001

 Date: 20/06/2025 11:54:09
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for book_category
-- ----------------------------
DROP TABLE IF EXISTS `book_category`;
CREATE TABLE `book_category`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '分类名称，如基础理论、算法实践',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 4 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of book_category
-- ----------------------------
INSERT INTO `book_category` VALUES (1, '初学者');
INSERT INTO `book_category` VALUES (2, '中级工程师');
INSERT INTO `book_category` VALUES (3, '高级研究者');

-- ----------------------------
-- Table structure for book_resource
-- ----------------------------
DROP TABLE IF EXISTS `book_resource`;
CREATE TABLE `book_resource`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '书籍名称',
  `storage_path` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件存储路径',
  `file_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件类型，如 PDF、EPUB',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '书籍简介',
  `audience` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '适用人群',
  `cover_path` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '封面图像路径',
  `category_id` int NOT NULL COMMENT '关联 book_category',
  `created_at` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT 'md内容',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_book_category`(`category_id` ASC) USING BTREE,
  FULLTEXT INDEX `ft_book_name_desc`(`name`, `description`),
  CONSTRAINT `fk_book_cat` FOREIGN KEY (`category_id`) REFERENCES `book_category` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 51 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of book_resource
-- ----------------------------
INSERT INTO `book_resource` VALUES (1, '区块链原理及其核心技术', '/books/book_1.pdf', 'PDF', '一种公司使用作为或者.这里而且帮助下载地方国家合作然后.\r\n音乐主题部门环境出现.最新你们组织评论您的.上海关于已经价格.\r\n处理谢谢人民要求可以知道一种.下载都是本站应该发展.\r\n数据服务那么发现时间通过制作.当前组织经营只要.\r\n还是合作一种不过更新服务报告您的.登录希望电话查看工具进行您的.类别继续能力组织密码得到.\r\n作为不过今天说明手机经验.\r\n实现两个自己.过程知道时候方法没有城市状态.不同这是资料应该回复问题.', '初学者', 'http://localhost:8080/profile/image/1748110285684.png', 2, '2025-01-11 09:17:52', NULL);
INSERT INTO `book_resource` VALUES (2, '计算机视觉技术 第2章', '/books/book_2.epub', 'EPUB', '说明工具方式.只要需要原因.\r\n点击可是发表增加.一下一般提高.\r\n应该提供在线提供来源.无法中文设计大小都是.一次学校选择根据自己游戏一些回复.\r\n那个对于类型免费支持.因此那些应该东西语言.\r\n主要都是留言密码出现可以她的.\r\n国际新闻基本全部.\r\n能力需要项目系列深圳不过中国.合作的是论坛原因怎么喜欢.\r\n必须系统使用今天方面都是介绍.企业标题市场组织精华.客户推荐论坛用户.\r\n游戏音乐因为觉得经验你们时间.无法当前大家为什只有网上.\r\n精华科技准备一直简介发表.有关任何其中加入技术.', '初学者', 'http://localhost:8080/profile/image/1748110285747.png', 3, '2024-06-13 14:06:40', NULL);
INSERT INTO `book_resource` VALUES (3, '大数据分析方法 第3章', '/books/book_3.pdf', 'PDF', '搜索他们评论解决其实到了经营.\r\n为了人民网站现在行业.\r\n需要其他当然看到关于表示功能今天.\r\n东西为了拥有建设日期其中.城市首页报告控制.留言什么以及这是.\r\n正在日本到了标准投资到了.会员到了科技.\r\n合作事情等级研究留言.进入重要之后完成自己经验资源.留言城市网站其实状态首页日期部门.分析系统以后感觉人员.', '中级工程师', 'http://localhost:8080/profile/image/1748110285918.png', 1, '2025-05-12 08:16:21', NULL);
INSERT INTO `book_resource` VALUES (4, '软件工程实践 第4章', '/books/book_4.pdf', 'PDF', '阅读建设不要我们之后一下记者虽然.喜欢标准其他软件之间专业孩子信息.\r\n学校帖子一定关于.现在制作会员两个如此那个.这个社会位置空间业务软件成功.\r\n自己地方由于.操作增加经验资源基本状态起来.\r\n深圳商品重要类型能力点击价格.主题解决出来规定进入制作.电子中心继续.空间显示比较只要影响.\r\n具有非常对于研究.企业电影感觉都是安全.\r\n时候来源注册来源.点击什么能力最大组织.\r\n还有世界成功以后帖子.能够市场喜欢开始经营注意.搜索电话人员个人手机系列.', '初学者', 'http://localhost:8080/profile/image/1748110286148.png', 3, '2024-07-09 21:06:57', NULL);
INSERT INTO `book_resource` VALUES (5, '机器学习算法与应用 第5章', '/books/book_5.pdf', 'PDF', '电子为什是一时间基本的人原因类别.那个谢谢那个留言空间项目手机.\r\n无法方式城市因为.都是一下能够为什活动合作.还是认为今天.\r\n什么提高规定还有广告.人民觉得社区.\r\n项目参加看到自己表示次数.孩子只是提高事情情况文章还是可以.\r\n社区已经通过最后可能.首页朋友登录具有电脑作为.\r\n孩子日本您的.我的开始有限数据.\r\n到了发布操作生活计划男人男人.不过这里大家如何什么.\r\n情况非常教育不是企业到了销售威望.经验地址项目类型如果历史.具有因此电话推荐建设价格.', '初学者', 'http://localhost:8080/profile/image/1748110286357.png', 2, '2025-01-22 17:22:11', NULL);
INSERT INTO `book_resource` VALUES (6, '云计算与分布式系统 第6章', '/books/book_6.pdf', 'PDF', '部门深圳他们表示.之后数据电话时候所以.次数可是中文学生是一的人.\r\n看到以及能力说明孩子参加.\r\n开始所以认为如果电脑一次.希望经济名称设计.首页数据不要图片.\r\n那些部门中心因此全国.搜索全部密码成为新闻只是如果.\r\n一些更多相关现在.最后特别如果电脑位置中文查看游戏.空间有限标准经营公司过程.\r\n活动直接各种联系手机东西参加提供.感觉分析论坛服务.\r\n一直等级图片建设那么工作这里位置.技术注意标题.\r\n必须事情城市那些部门使用那么分析.为什发展系统孩子时候今天要求.朋友对于方法喜欢论坛精华.', '高级研究者', 'http://localhost:8080/profile/image/1748110287079.png', 1, '2025-03-14 05:58:17', NULL);
INSERT INTO `book_resource` VALUES (7, '自然语言处理实战 第7章', '/books/book_7.epub', 'EPUB', '合作所以提供都是她的论坛.本站环境用户阅读一下使用积分.\r\n实现方法中国的话是否制作.也是一样搜索发生政府.\r\n学生品牌上海开始只是标准浏览.世界地方完全使用虽然.为什发展留言本站最后.显示其他来源今天阅读包括.\r\n影响回复但是地方发现免费个人.功能教育出来一些.\r\n电脑现在程序.音乐到了城市发生.\r\n大小正在国内起来点击.能力世界网络部门出现网上政府.\r\n一样发现大学以下合作.基本起来用户.不是法律网络人员原因部分无法.', '中级工程师', 'http://localhost:8080/profile/image/1748110287666.png', 2, '2024-11-24 14:10:44', NULL);
INSERT INTO `book_resource` VALUES (8, '软件工程实践 第8章', '/books/book_8.epub', 'EPUB', '搜索地址有关在线只要.\r\n教育情况目前其实得到大学.技术为了只是不过工程你们包括会员.\r\n其他可以游戏拥有网站政府准备.投资次数运行不是产品.位置直接电子那么.\r\n特别或者关系操作人员位置.国内帮助直接关系介绍这样这是.起来一个更新认为信息.\r\n阅读状态推荐.之后一种时候她的.\r\n时候是一首页网络因为.资料而且全国.', '中级工程师', 'http://localhost:8080/profile/image/1748110288016.png', 1, '2024-10-11 22:26:19', NULL);
INSERT INTO `book_resource` VALUES (9, '人工智能的未来 第9章', '/books/book_9.pdf', 'PDF', '因此直接成为专业要求.如此设备游戏的是作为时间什么.\r\n学校特别软件国家各种教育.大学公司网站.提高规定完全帮助都是.\r\n简介根据时候而且今天内容.由于登录政府提高提高.\r\n只有同时环境.行业一般专业当前文件回复网上.经济主题一直包括密码.表示设计不是学习次数处理.\r\n全国国内具有那么结果因为这么.不同图片企业免费其中功能市场.合作专业这是可能使用正在介绍.\r\n成功控制不同来自.密码可以一般电脑实现控制商品首页.留言关于开发感觉一定.\r\n联系说明功能可是主要一个一样.能够进行工程所以.事情生活或者进入电脑.', '中级工程师', 'http://localhost:8080/profile/image/1748110288301.png', 1, '2024-08-26 10:38:44', NULL);
INSERT INTO `book_resource` VALUES (10, '数据科学基础 第10章', '/books/book_10.epub', 'EPUB', '帮助电脑发布为了认为.\r\n这是有关没有.当然原因不能分析资料.\r\n登录没有如此直接一定价格历史.质量但是操作电脑简介汽车.\r\n两个一定这是结果希望两个当前.以后必须报告投资生活行业方面.要求情况全国开发日本希望.\r\n希望品牌谢谢起来觉得起来名称.合作大小国内情况有限经济学校.', '高级研究者', 'http://localhost:8080/profile/image/1748110288637.png', 3, '2025-01-25 00:04:45', NULL);
INSERT INTO `book_resource` VALUES (11, '自然语言处理实战 第11章', '/books/book_11.pdf', 'PDF', '非常社区可以资源.孩子继续拥有责任的话只有一种.\r\n喜欢回复教育论坛.这是控制时间发表应该为了评论.大小其他留言认为汽车最大这些.电话业务以后合作产品中文帖子任何.\r\n数据位置汽车不断能够状态这种.比较准备论坛空间.资源人民网上不要之间觉得教育.\r\n知道发展最大他们.如何中文注册发展活动.加入历史音乐.', '初学者', 'http://localhost:8080/profile/image/1748110288783.jpg', 1, '2025-04-07 22:26:24', NULL);
INSERT INTO `book_resource` VALUES (12, '自然语言处理实战 第12章', '/books/book_12.epub', 'EPUB', '计划浏览提供现在.次数那么简介深圳认为类型.或者虽然密码详细今年图片.\r\n服务谢谢出现那么需要学生点击.客户他们这个最后注册.\r\n什么政府图片科技特别各种.质量制作详细作为.操作结果点击浏览直接学生.管理完成地区北京操作.\r\n公司有些其实可以当前一定出现.最大还是继续注册文件结果公司.\r\n回复通过也是.\r\n内容实现文化下载电影品牌.\r\n解决主题计划类型销售网上企业.提供工作设计.两个行业准备的是一般部门.', '中级工程师', 'http://localhost:8080/profile/image/1748110288796.png', 3, '2025-01-21 00:59:20', NULL);
INSERT INTO `book_resource` VALUES (13, '自然语言处理实战 第13章', '/books/book_13.pdf', 'PDF', '在线女人历史这些.其实销售合作现在日期.\r\n项目一般拥有也是什么一般什么.大小正在简介生活空间觉得注册当然.\r\n当然不同最新文化增加然后制作.自己不会以后精华一直增加.应用主题发生地址规定方式.\r\n文件欢迎以及电影商品登录所以.市场什么工具女人搜索责任.\r\n生活那个注册服务继续登录.服务阅读包括软件.', '高级研究者', 'http://localhost:8080/profile/image/1748110288816.png', 1, '2025-04-17 13:30:33', NULL);
INSERT INTO `book_resource` VALUES (14, '人工智能的未来 第14章', '/books/book_14.pdf', 'PDF', '拥有一种还有报告研究国内名称.法律有关科技系统电影今天.信息原因操作会员地址进行您的.\r\n得到这些能够工程注意搜索标题.文章首页全国中国能力.网上都是他们.\r\n精华上海社区用户.通过发布说明帮助发表作者.\r\n方法免费原因类别质量本站.法律两个拥有服务.\r\n具有其他情况.成为部分系统感觉技术.这个决定控制中国任何详细阅读什么.', '中级工程师', 'http://localhost:8080/profile/image/1748110289041.png', 3, '2024-06-26 20:49:52', NULL);
INSERT INTO `book_resource` VALUES (15, '软件工程实践 第15章', '/books/book_15.epub', 'EPUB', '网络今天有限来源社区.以上朋友品牌的人还是自己管理.为了一次重要方面位置中文.\r\n各种技术更新解决包括.提高计划得到更多现在这种不能数据.\r\n记者朋友你们其他上海搜索地区.留言学生关于的话产品.国家使用公司.业务一直看到软件责任方面有限.', '中级工程师', 'http://localhost:8080/profile/image/book-placeholder.jpg', 1, '2025-01-14 01:29:34', NULL);
INSERT INTO `book_resource` VALUES (16, '自然语言处理实战 第16章', '/books/book_16.epub', 'EPUB', '环境必须本站.不是增加就是作者谢谢世界计划.\r\n大小进入搜索得到如果最后时间.空间以上用户.人员怎么能力她的类别直接.\r\n一定的人那个.其中得到我们实现.\r\n如此地址质量电脑免费.以后增加东西地方计划.\r\n电话欢迎投资以及设备类别.事情学生决定学习觉得之间.新闻学校这种发表因此您的来自更新.\r\n说明设计国际说明专业.之间当然管理能够显示日期.\r\n作者朋友准备成功操作国际.上海本站教育人民.密码这个就是地址结果比较感觉项目.\r\n经验到了价格到了觉得音乐作品.比较工作国家等级简介其他部分.', '中级工程师', 'http://localhost:8080/profile/image/搜图神器_1748110121066.png', 2, '2025-05-03 16:51:09', NULL);
INSERT INTO `book_resource` VALUES (17, '数据科学基础 第17章', '/books/book_17.epub', 'EPUB', '无法到了任何一个规定点击进入日期.因此是一不过价格密码的是通过.\r\n最后关系信息活动各种帖子.业务汽车不会分析.\r\n大学法律这个分析资源产品最新.游戏只是完成知道公司有关联系.到了一起怎么.\r\n什么学习回复中心.已经功能只是详细.销售运行那个销售.\r\n手机图片比较.', '高级研究者', 'http://localhost:8080/profile/image/搜图神器_1748110124591.png', 3, '2025-04-10 22:14:45', NULL);
INSERT INTO `book_resource` VALUES (18, '深度学习入门 第18章', '/books/book_18.pdf', 'PDF', '企业管理只要其实只是规定以及.网上搜索必须专业电影其实.只要经验准备能力经济问题.\r\n企业男人介绍或者完全.可是完全完成参加.\r\n继续不过由于.详细实现注意.\r\n因为工具作为所以.研究不能商品销售网络.\r\n可以时候一直.生产注意特别以下不要.\r\n希望中心目前国内.进入日本发布搜索完全投资注意.文化你们开始生活程序最新.\r\n特别喜欢图片一点下载人员那么.认为根据而且大家要求自己.\r\n影响注册欢迎最大位置所有你的.影响显示根据.女人一下地址起来.精华生产帮助可以来自.', '中级工程师', 'http://localhost:8080/profile/image/搜图神器_1748110137527.png', 1, '2024-11-23 12:41:25', NULL);
INSERT INTO `book_resource` VALUES (19, '自然语言处理实战 第19章', '/books/book_19.epub', 'EPUB', '自己重要文章电话.支持自己同时一种免费.\r\n要求非常出现状态经济.感觉您的大小产品重要任何.价格结果联系教育.以上更新市场各种一直价格注册是一.\r\n国家如何项目说明说明你的.大小部分说明.\r\n登录美国各种市场业务投资.来源销售这样.数据而且能力精华更新积分下载.不会其中情况投资次数提高本站.\r\n评论中文自己成为更多国际地址.积分方面开发电话专业自己.\r\n推荐希望注册电话.一般计划特别虽然.国内中国已经能够这些.过程用户继续目前是一详细.\r\n回复首页虽然功能来源当然时间.国际重要虽然出来设计.', '初学者', 'http://localhost:8080/profile/image/搜图神器_1748110141476.png', 3, '2024-12-16 15:19:10', NULL);
INSERT INTO `book_resource` VALUES (20, '自然语言处理实战 第20章', '/books/book_20.pdf', 'PDF', '产品地址之间系列威望.商品安全这个的是企业.\r\n制作下载名称详细.进行最新网上规定选择活动过程.工具一个那个为什如此.\r\n教育这么任何作为法律比较一下方法.什么名称大学一定法律.\r\n基本学习作者结果国内起来查看一切.看到威望结果安全政府.', '初学者', NULL, 2, '2024-08-16 06:44:35', NULL);
INSERT INTO `book_resource` VALUES (21, '深度学习入门 第21章', '/books/book_21.pdf', 'PDF', '日本非常工程等级.法律虽然通过企业服务文件中心.\r\n学习必须帮助计划工具组织对于.方面什么同时虽然虽然比较.关系信息还是.\r\n作品最后您的这么你们不能.说明你们你们已经不断.\r\n项目不是提高作者北京以及.技术状态环境社会有关.更多的话他的准备已经然后完全.\r\n留言当然发表用户无法所以也是.行业男人应用现在这个.\r\n以上今天不断要求政府都是技术精华.地方非常发现以下部分.登录信息评论发布这样最后学习.\r\n可能业务事情一直不断作品.可以商品详细一定各种因此可能.', '初学者', NULL, 2, '2024-10-07 17:41:46', NULL);
INSERT INTO `book_resource` VALUES (22, '区块链技术详解 第22章', '/books/book_22.epub', 'EPUB', '浏览发布世界不断.都是运行一个只是.\r\n这种来自由于积分主要因此功能.投资网上自己在线.\r\n过程登录原因的人人员学校这是工具.也是生活文件通过以下如此支持.\r\n学习关于使用有关.其他专业已经如此密码图片.他的影响可以图片非常.免费项目时候或者进行系列.', '中级工程师', NULL, 1, '2025-05-21 21:52:58', NULL);
INSERT INTO `book_resource` VALUES (23, '区块链技术详解 第23章', '/books/book_23.pdf', 'PDF', '生活功能文件的是只是电话.喜欢为什位置上海出来状态本站.\r\n怎么然后记者一起工程人民我们.而且主要因此.大学一切记者女人.\r\n表示处理孩子任何程序.国家生产已经就是.\r\n关系两个其他论坛所以一些那个.\r\n项目首页主要我的介绍这个成为.学习表示你们.\r\n世界今天专业支持觉得今年.注意日期科技完成.如何提供论坛解决次数完成.\r\n通过不能软件出来.专业之间可是不同功能组织.说明进入作者汽车.', '中级工程师', '/covers/cover_23.jpg', 3, '2024-11-13 17:04:22', NULL);
INSERT INTO `book_resource` VALUES (24, '数据科学基础 第24章', '/books/book_24.epub', 'EPUB', '建设语言手机环境更多正在.可以点击市场目前.可是报告安全.\r\n什么行业关系这样网站东西.是否网络方法会员男人一点控制.大学工作经验行业行业.图片不过说明运行研究.\r\n不断必须学习一下.\r\n决定为什一样增加当然.分析影响问题不同本站销售.不是地址只是完成根据.的话比较简介操作由于地址他的时候.\r\n其他上海公司应该可以.特别教育功能.或者汽车之间感觉积分.\r\n世界电影显示时间内容.实现产品首页孩子他们以上.就是发现所有什么类型.', '高级研究者', NULL, 1, '2024-11-24 21:08:58', NULL);
INSERT INTO `book_resource` VALUES (25, '大数据分析方法 第25章', '/books/book_25.pdf', 'PDF', '北京语言电话主题一种研究.名称这么目前一次.工作商品公司影响.\r\n正在那么管理能力由于.这里搜索城市应用有限他的以后.主要之间出现.\r\n商品地方更多计划只有的人.当前程序有关经验服务.\r\n注意本站之后以后自己完全对于.', '高级研究者', NULL, 2, '2025-02-16 01:06:18', NULL);
INSERT INTO `book_resource` VALUES (26, '数据科学基础 第26章', '/books/book_26.pdf', 'PDF', '完全但是一次世界学习他的分析.产品工程工作地方.\r\n人民能够首页评论活动.公司科技产品发布控制合作使用.只要他的更新然后.\r\n也是希望会员各种.\r\n国内电影专业影响管理发表推荐.作者一样直接选择电子重要直接.安全但是以后网站直接.\r\n朋友作品信息分析威望市场只有.免费原因注意登录进入留言.政府音乐自己不同首页评论电脑.活动网络网上生活作者的话.\r\n一起图片产品推荐行业得到回复.朋友设计美国女人继续介绍.\r\n已经知道结果图片其实这样信息.非常解决没有一个孩子有限发表所以.那些主要一个很多空间为什.没有一样类别销售由于都是学校.', '高级研究者', NULL, 3, '2024-06-29 20:21:48', NULL);
INSERT INTO `book_resource` VALUES (27, '机器学习算法与应用 第27章', '/books/book_27.epub', 'EPUB', '结果语言论坛发表.支持无法说明以后学习自己.\r\n汽车说明技术为了.国际显示其中出来.喜欢怎么原因地址.\r\n增加电影怎么工作就是软件电影.工具功能日期进行.广告一个成为工程.\r\n地区个人评论活动还是学校.计划一种得到美国语言商品实现.觉得国家活动特别表示.非常用户其他文章.\r\n网络企业男人大学有些安全虽然这样.男人选择为了单位注意有限大家所有.包括时候城市.', '中级工程师', '/covers/cover_27.jpg', 1, '2025-01-07 01:45:57', NULL);
INSERT INTO `book_resource` VALUES (28, '自然语言处理实战 第28章', '/books/book_28.pdf', 'PDF', '文化产品游戏喜欢经营应该.经济什么支持工具解决之后类型这种.知道销售积分发布当前.\r\n时间密码结果特别具有提供非常.任何空间注册的是欢迎一种浏览.今天可以其他作为之后.\r\n会员功能发布下载图片.您的不断孩子朋友客户经营.\r\n拥有主题市场社会注意.', '初学者', NULL, 2, '2025-01-25 22:13:58', NULL);
INSERT INTO `book_resource` VALUES (29, '自然语言处理实战 第29章', '/books/book_29.epub', 'EPUB', '标题中国你的当前今年国际文化.类型发表时候准备准备关系具有.\r\n游戏今年记者文化合作.而且控制最新功能合作那些以上.最大运行欢迎朋友.\r\n朋友一点音乐公司应该.空间资源程序提供以后.程序专业全国因为现在免费.\r\n威望决定商品那个经营部分一种.发布制作那么类别可是可是通过.\r\n以及欢迎积分但是可以.时候科技不过国内标题其中.', '中级工程师', NULL, 2, '2025-04-30 21:04:04', NULL);
INSERT INTO `book_resource` VALUES (30, '大数据分析方法 第30章', '/books/book_30.pdf', 'PDF', '东西电子主题评论业务发生.\r\n一直为了男人方面.进入环境完全喜欢程序.觉得自己支持活动记者时间.\r\n完全能够系列帮助来源类型精华参加.\r\n情况公司因此加入类别.活动因此类型而且.\r\n不能注意那么产品最大资料.网上这些都是.通过无法单位这个功能有关也是文章.\r\n人民大学没有.有关还是帖子主要参加国际提高时候.\r\n游戏很多更多政府.等级正在规定密码服务免费表示.\r\n会员男人必须经济她的特别名称.进行上海密码不过注意就是单位.都是来自经济.\r\n销售电话的话大小直接表示其中.这么关于时候文章新闻信息美国.增加他们城市全国合作出来然后.\r\n你的其实市场.点击设备完全价格.两个实现只要解决开始记者.', '中级工程师', NULL, 1, '2024-09-25 14:56:23', NULL);
INSERT INTO `book_resource` VALUES (31, '深度学习入门 第31章', '/books/book_31.epub', 'EPUB', '发表技术电脑科技关系.可以只要觉得运行部门网站以上.在线虽然比较参加制作.\r\n方面图片搜索最新留言一般国家.学习或者一直.之间销售推荐部分.文章更多作为文化分析.\r\n出现发生准备虽然实现历史如果.合作相关等级联系详细感觉.更多语言北京回复当然.\r\n有关男人个人产品汽车类别.\r\n所以运行出来一定电脑.方式新闻帖子今年主要影响非常.这是特别包括点击.\r\n之后学习更多对于还有.产品方面觉得系统.', '中级工程师', NULL, 1, '2024-10-22 19:15:39', NULL);
INSERT INTO `book_resource` VALUES (32, '人工智能的未来 第32章', '/books/book_32.epub', 'EPUB', '手机增加对于提供.法律图片作为然后搜索有关.电影所以起来解决.\r\n帮助重要公司更多来自全国.建设一般帮助环境位置显示人员.\r\n不能不能积分精华.威望的人网络威望当然.\r\n成为系统中文浏览根据登录.作者其实市场成为这个实现更多直接.', '高级研究者', NULL, 3, '2025-05-24 02:55:11', NULL);
INSERT INTO `book_resource` VALUES (33, '自然语言处理实战 第33章', '/books/book_33.pdf', 'PDF', '系统通过方式研究一起图片.学生设计实现他的大家.\r\n包括对于事情经验工程也是设备更多.中文提高科技状态世界本站您的.社区增加公司必须影响如何这些.\r\n包括回复不能自己我的主题.\r\n注册主要城市都是专业.环境今天自己有关操作具有报告.', '初学者', '/covers/cover_33.jpg', 3, '2024-09-05 11:05:10', NULL);
INSERT INTO `book_resource` VALUES (34, '计算机视觉技术 第34章', '/books/book_34.epub', 'EPUB', '专业包括非常能力.大小得到开发然后.广告增加你的更新的话虽然学生一定.\r\n方式应该根据标准类型相关.选择部门合作一样方面生活.电影很多国内.\r\n他们可以是否时间方面数据所有汽车.很多报告提供.所有部分广告东西.', '中级工程师', '/covers/cover_34.jpg', 1, '2024-10-19 04:58:05', NULL);
INSERT INTO `book_resource` VALUES (35, '深度学习入门 第35章', '/books/book_35.pdf', 'PDF', '电子无法能够政府学生.信息发生大小有限需要网站表示.\r\n首页日本虽然点击合作这个大小.能力感觉活动活动.人员全部男人电脑大学就是留言.公司大家控制内容然后所以.\r\n设备这里帮助分析功能孩子.这个活动浏览作品分析浏览过程.特别实现虽然等级关系孩子很多分析.地址事情服务通过决定同时.\r\n状态只有点击更新全部但是选择.女人新闻中心等级之后.\r\n单位社会点击到了知道谢谢不过.记者当前男人质量.\r\n目前进入阅读部门更多今天可是国家.情况包括网站.\r\n以及下载推荐我们.必须这个资料研究.国际积分开发有关今天日本.\r\n发生类别管理内容控制点击可以学习.过程新闻应该出来问题.', '高级研究者', NULL, 3, '2024-09-25 07:23:46', NULL);
INSERT INTO `book_resource` VALUES (36, '云计算与分布式系统 第36章', '/books/book_36.epub', 'EPUB', '文章能力相关组织市场电子来源全部.\r\n应用其实通过.关系主要时候正在评论设计.谢谢能力空间程序具有帖子.\r\n今天你们有关其他所有同时规定.目前信息说明行业中心.\r\n类别孩子出来感觉.简介必须活动规定您的就是.\r\n得到时间要求什么.电子活动服务投资朋友资源以及.这样提供中国论坛.\r\n那个结果介绍那个就是完成.一下日本情况推荐.\r\n继续大家一点所有或者通过过程.单位记者等级这么.次数到了方式不同的是这是.', '中级工程师', NULL, 2, '2024-09-23 21:33:26', NULL);
INSERT INTO `book_resource` VALUES (37, '数据科学基础 第37章', '/books/book_37.pdf', 'PDF', '看到数据介绍喜欢介绍.而且下载出来.合作市场全国因为公司.信息文章有限专业能力.\r\n提高留言搜索一直那个这种虽然.注册以上合作问题情况密码网站.项目更新文化一些要求一直.\r\n发表孩子方式我的我的.一个服务资源免费具有业务图片.回复女人她的谢谢基本关系.\r\n品牌因此对于社区位置也是.决定任何发现大小.不能游戏服务内容今天主要.\r\n销售然后喜欢投资最后人民.工作进行设计.', '高级研究者', NULL, 1, '2024-08-12 11:18:31', NULL);
INSERT INTO `book_resource` VALUES (38, '大数据分析方法 第38章', '/books/book_38.pdf', 'PDF', '感觉中文决定类别因为女人.介绍主题来自来自谢谢制作提高.问题开发空间工具工具必须城市.\r\n有关服务品牌当然说明只是系列.文件记者为什具有中文看到继续.音乐上海以上觉得.网站解决如何开发报告.', '中级工程师', NULL, 1, '2024-10-22 18:23:46', NULL);
INSERT INTO `book_resource` VALUES (39, '数据科学基础 第39章', '/books/book_39.pdf', 'PDF', '发表下载北京之间这是人员.\r\n结果应该一切有关完成部门更新.一切历史中国.关于准备这种责任.\r\n以下不断推荐那些.一点一定如何你的完成.表示他们应用以及计划其实密码部分.', '中级工程师', NULL, 2, '2025-03-20 07:52:06', NULL);
INSERT INTO `book_resource` VALUES (40, '云计算与分布式系统 第40章', '/books/book_40.pdf', 'PDF', '威望名称时候责任.发展参加根据今天.\r\n活动对于我的有些这些这种首页.环境客户没有一样工作.\r\n解决不会我的首页目前.设备制作地址上海中国.\r\n我的积分成功设备信息工作能够.之后包括其他程序怎么你的.\r\n运行应用深圳系统女人原因世界.解决经验网上研究不是你们方面.操作更多软件经济重要东西处理.\r\n登录公司可是论坛信息的人.销售她的其实自己显示.浏览需要公司这是.数据方面名称单位.\r\n由于时间一起品牌.资料以下论坛业务希望提高.\r\n社区发布投资内容联系同时.', '初学者', NULL, 1, '2024-06-25 01:08:03', NULL);
INSERT INTO `book_resource` VALUES (41, '云计算与分布式系统 第41章', '/books/book_41.pdf', 'PDF', '项目新闻留言一定专业无法这些.计划通过根据.\r\n还是报告免费大学.\r\n支持是一得到准备教育位置.因为怎么浏览网络单位.\r\n以后出来为了发生世界.以上音乐用户一个国际.\r\n的是合作成为其他.说明数据名称那些操作他的支持.\r\n包括认为因为空间其他很多回复只是.\r\n也是音乐为了学校质量处理技术.能力这些北京法律全国还有.出来法律时候国家资料.', '中级工程师', '/covers/cover_41.jpg', 3, '2024-08-15 15:40:02', NULL);
INSERT INTO `book_resource` VALUES (42, '人工智能的未来 第42章', '/books/book_42.epub', 'EPUB', '空间表示显示本站.规定功能谢谢现在中文信息方式发表.\r\n网站在线一样他们一切他的日本商品.介绍一切是否安全美国.不要都是论坛时间日期.\r\n中国作为科技重要经验.大学不同软件需要.必须起来不过主要虽然她的过程.\r\n控制能够原因北京法律合作.但是我们历史.环境手机介绍组织表示详细电脑.\r\n然后直接不过之间其中因此.\r\n感觉用户具有参加下载.规定或者留言这个名称.现在其他因此学习具有可以不是.\r\n本站非常的话部分以后.个人城市那些提高进行一直.人民一切也是发表那么标准人民位置.社区自己技术开始这么生活社区.\r\n浏览过程报告完全北京.目前全部使用你们.', '初学者', NULL, 2, '2025-05-21 01:29:02', NULL);
INSERT INTO `book_resource` VALUES (43, '人工智能的未来 第43章', '/books/book_43.pdf', 'PDF', '人民报告美国时候部分.之间网上公司一样.女人简介显示那些网络要求.\r\n男人开发组织使用不同程序音乐.我们方面密码继续推荐的话她的他的.所有操作其实人员这种销售.\r\n由于设计只要销售比较那个工程.谢谢任何历史评论法律评论增加认为.不能怎么的话生活谢谢这种.\r\n最大文化进入这么地方搜索单位.准备这些都是方式今年加入地区.', '高级研究者', NULL, 3, '2025-02-04 07:19:44', NULL);
INSERT INTO `book_resource` VALUES (44, '大数据分析方法 第44章', '/books/book_44.epub', 'EPUB', '的是教育关于人员决定行业生产有些.\r\n同时应该为了一样.标题希望项目详细而且上海方法.网上论坛使用研究.\r\n作者而且威望他们.决定状态就是还有搜索.社区发现选择制作密码成为.\r\n学生根据一起完成知道地区关于.首页今天进行单位因为本站进行.下载深圳起来如果具有电影.\r\n到了还有由于不要最后.那些文化服务以及可能.\r\n用户不会中文研究的是可以.还是历史中文工具.本站世界作者最新制作男人以下.\r\n不要如何发现而且音乐今天资料问题.朋友控制那些类别然后.\r\n工具参加情况.状态还有增加文章同时已经.', '初学者', '/covers/cover_44.jpg', 1, '2025-02-07 00:09:01', NULL);
INSERT INTO `book_resource` VALUES (45, '数据科学基础 第45章', '/books/book_45.epub', 'EPUB', '能力女人网络这样情况.服务一般次数这些企业如何或者继续.\r\n查看国家首页功能一点合作一种.你们的是电话部门安全中国.\r\n也是公司需要科技.是否自己之后自己工作地区登录增加.当然公司游戏解决当前事情对于.\r\n价格公司这些质量如果规定如何.增加她的游戏社区的人成功得到威望.\r\n类别人民特别市场所有朋友.信息工具虽然是一设计他的用户.都是环境客户最大音乐.\r\n现在本站会员更多有些重要.\r\n地方男人公司完全.网络积分业务没有个人您的系统.能够一起主要本站帖子世界希望.', '初学者', NULL, 1, '2024-06-30 04:30:50', NULL);
INSERT INTO `book_resource` VALUES (46, '软件工程实践 第46章', '/books/book_46.epub', 'EPUB', '出来欢迎显示使用朋友查看.设计不断这么大家文章电话中文.\r\n国际能力一些规定帮助.之间企业影响空间相关原因那个.关系发展安全但是品牌浏览我们.\r\n资料软件你们现在安全电影介绍.\r\n自己显示生活教育结果这么.朋友点击特别正在深圳这些.\r\n科技联系还是运行.\r\n联系原因汽车报告说明其他.基本有限这里人员.\r\n发展市场没有合作.实现这个人民单位.', '初学者', '/covers/cover_46.jpg', 1, '2024-12-07 16:36:31', NULL);
INSERT INTO `book_resource` VALUES (47, '计算机视觉技术 第47章', '/books/book_47.pdf', 'PDF', '生活用户产品控制精华使用.能力拥有非常操作回复一般.什么社区点击基本教育为了.\r\n能力必须提供很多这里我们责任.欢迎她的资源.\r\n部门根据使用记者特别过程内容.感觉一下时间使用标准一种.语言大学论坛软件的是.\r\n如果关于类别全国.现在组织有限工具.软件软件可以不断类型社会.\r\n因为能力免费积分继续中文.\r\n音乐中国他们一直女人回复参加.公司责任北京中文部分表示.\r\n最新需要广告一切企业他的已经.如何方法就是上海.威望地区个人电脑搜索东西.', '初学者', NULL, 1, '2024-09-25 04:12:51', NULL);
INSERT INTO `book_resource` VALUES (48, '自然语言处理实战 第48章', '/books/book_48.epub', 'EPUB', '产品只要这个空间标准大小自己.大学男人任何同时.\r\n威望投资而且虽然加入增加广告当前.\r\n历史企业因此注意.对于当前客户这些虽然发现.建设影响女人一定.\r\n注意自己注册一直需要.方面功能不会为什销售.\r\n基本简介最大人民.\r\n应该人民原因设备.', '中级工程师', '/covers/cover_48.jpg', 2, '2025-01-26 14:26:39', NULL);
INSERT INTO `book_resource` VALUES (49, '人工智能的未来 第49章', '/books/book_49.pdf', 'PDF', '还是公司现在.\r\n浏览根据发布语言工程市场.男人不能市场客户开发事情全部.\r\n问题运行因此上海只有但是.主题两个在线其实价格.\r\n有些主题以后关系过程.文章经验运行责任不断报告各种.资源法律投资看到国际这么.就是中文处理女人关于的话还是中文.\r\n能力增加一下文章他们.男人功能工作安全日期企业.当前阅读以后来自文章进行精华.\r\n必须原因成功.今年必须名称东西会员全国.', '初学者', '/covers/cover_49.jpg', 3, '2025-03-11 16:39:58', NULL);
INSERT INTO `book_resource` VALUES (50, '大数据分析方法 第50章', '/books/book_50.pdf', 'PDF', '你们首页合作日期不过他的.系统之间北京论坛.无法不是系列方式喜欢他的为什.希望功能游戏对于价格科技.\r\n不会资源或者到了起来不同标准.位置公司无法操作状态深圳.\r\n更新为什而且教育的是.必须需要电脑然后搜索.\r\n生产管理经济论坛.全部帮助我的实现通过问题国内情况.拥有之后程序经营帮助.\r\n人员如果的话因为.\r\n科技威望文化当然其实.她的必须网站其他拥有.\r\n规定操作或者工程.相关可是一种行业管理.一种怎么方法不是密码当然.\r\n生产美国分析女人城市数据.为了主要这些注册需要自己质量.', '高级研究者', NULL, 1, '2024-07-14 05:03:49', NULL);

-- ----------------------------
-- Table structure for gen_table
-- ----------------------------
DROP TABLE IF EXISTS `gen_table`;
CREATE TABLE `gen_table`  (
  `table_id` bigint NOT NULL AUTO_INCREMENT COMMENT '编号',
  `table_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '表名称',
  `table_comment` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '表描述',
  `sub_table_name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '关联子表的表名',
  `sub_table_fk_name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '子表关联的外键名',
  `class_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '实体类名称',
  `tpl_category` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT 'crud' COMMENT '使用的模板（crud单表操作 tree树表操作）',
  `tpl_web_type` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '前端模板类型（element-ui模版 element-plus模版）',
  `package_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '生成包路径',
  `module_name` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '生成模块名',
  `business_name` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '生成业务名',
  `function_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '生成功能名',
  `function_author` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '生成功能作者',
  `gen_type` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '生成代码方式（0zip压缩包 1自定义路径）',
  `gen_path` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '/' COMMENT '生成路径（不填默认项目路径）',
  `options` varchar(1000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '其它生成选项',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`table_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 5 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '代码生成业务表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of gen_table
-- ----------------------------
INSERT INTO `gen_table` VALUES (1, 'book_category', '图书分类', NULL, NULL, 'BookCategory', 'crud', 'element-plus', 'com.ruoyi.cms', 'cms', 'book_category', '图书分类', 'lhl', '0', '/', '{\"parentMenuId\":2001}', 'admin', '2025-05-20 16:43:14', '', '2025-05-21 18:01:47', NULL);
INSERT INTO `gen_table` VALUES (2, 'book_resource', '图书', NULL, NULL, 'BookResource', 'crud', 'element-plus', 'com.ruoyi.cms', 'cms', 'book', '图书', 'lhl', '0', '/', '{\"parentMenuId\":2052}', 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51', NULL);
INSERT INTO `gen_table` VALUES (3, 'video_category', '视频分类', NULL, NULL, 'VideoCategory', 'crud', 'element-plus', 'com.ruoyi.cms', 'cms', 'video_category', '视频分类', 'lhl', '0', '/', '{\"parentMenuId\":2002}', 'admin', '2025-05-20 16:43:14', '', '2025-05-21 18:01:58', NULL);
INSERT INTO `gen_table` VALUES (4, 'video_resource', '视频', NULL, NULL, 'VideoResource', 'crud', 'element-plus', 'com.ruoyi.cms', 'cms', 'video', '视频', 'lhl', '0', '/', '{\"parentMenuId\":2052}', 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43', NULL);

-- ----------------------------
-- Table structure for gen_table_column
-- ----------------------------
DROP TABLE IF EXISTS `gen_table_column`;
CREATE TABLE `gen_table_column`  (
  `column_id` bigint NOT NULL AUTO_INCREMENT COMMENT '编号',
  `table_id` bigint NULL DEFAULT NULL COMMENT '归属表编号',
  `column_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '列名称',
  `column_comment` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '列描述',
  `column_type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '列类型',
  `java_type` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT 'JAVA类型',
  `java_field` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT 'JAVA字段名',
  `is_pk` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否主键（1是）',
  `is_increment` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否自增（1是）',
  `is_required` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否必填（1是）',
  `is_insert` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否为插入字段（1是）',
  `is_edit` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否编辑字段（1是）',
  `is_list` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否列表字段（1是）',
  `is_query` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否查询字段（1是）',
  `query_type` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT 'EQ' COMMENT '查询方式（等于、不等于、大于、小于、范围）',
  `html_type` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '显示类型（文本框、文本域、下拉框、复选框、单选框、日期控件）',
  `dict_type` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '字典类型',
  `sort` int NULL DEFAULT NULL COMMENT '排序',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`column_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 23 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '代码生成业务表字段' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of gen_table_column
-- ----------------------------
INSERT INTO `gen_table_column` VALUES (1, 1, 'id', NULL, 'int', 'Long', 'id', '1', '1', '0', '1', NULL, NULL, NULL, 'EQ', 'input', '', 1, 'admin', '2025-05-20 16:43:14', '', '2025-05-21 18:01:47');
INSERT INTO `gen_table_column` VALUES (2, 1, 'name', '分类名称，如基础理论、算法实践', 'varchar(100)', 'String', 'name', '0', '0', '1', '1', '1', '1', '1', 'LIKE', 'input', '', 2, 'admin', '2025-05-20 16:43:14', '', '2025-05-21 18:01:47');
INSERT INTO `gen_table_column` VALUES (3, 2, 'id', '', 'int', 'Long', 'id', '1', '1', '0', '1', NULL, NULL, NULL, 'EQ', 'input', '', 1, 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51');
INSERT INTO `gen_table_column` VALUES (4, 2, 'name', '书籍名称', 'varchar(200)', 'String', 'name', '0', '0', '1', '1', '1', '1', '1', 'LIKE', 'input', '', 2, 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51');
INSERT INTO `gen_table_column` VALUES (5, 2, 'storage_path', '文件存储路径', 'varchar(500)', 'String', 'storagePath', '0', '0', '1', '1', '1', '1', '1', 'EQ', 'textarea', '', 3, 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51');
INSERT INTO `gen_table_column` VALUES (6, 2, 'file_type', '文件类型，如 PDF、EPUB', 'varchar(50)', 'String', 'fileType', '0', '0', '1', '1', '1', '1', '1', 'EQ', 'select', '', 4, 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51');
INSERT INTO `gen_table_column` VALUES (7, 2, 'description', '书籍简介', 'text', 'String', 'description', '0', '0', '0', '1', '1', '1', '1', 'EQ', 'textarea', '', 5, 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51');
INSERT INTO `gen_table_column` VALUES (8, 2, 'audience', '适用人群', 'varchar(100)', 'String', 'audience', '0', '0', '0', '1', '1', '1', '1', 'EQ', 'input', '', 6, 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51');
INSERT INTO `gen_table_column` VALUES (9, 2, 'category_id', '关联 book_category', 'int', 'Long', 'categoryId', '0', '0', '1', '1', '1', '1', '1', 'EQ', 'input', '', 8, 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51');
INSERT INTO `gen_table_column` VALUES (10, 2, 'created_at', '', 'datetime', 'Date', 'createdAt', '0', '0', '0', '0', '0', '0', '0', 'EQ', 'datetime', '', 9, 'admin', '2025-05-20 16:43:14', '', '2025-05-25 02:52:51');
INSERT INTO `gen_table_column` VALUES (11, 3, 'id', NULL, 'int', 'Long', 'id', '1', '1', '0', '1', NULL, NULL, NULL, 'EQ', 'input', '', 1, 'admin', '2025-05-20 16:43:14', '', '2025-05-21 18:01:58');
INSERT INTO `gen_table_column` VALUES (12, 3, 'name', '分类名称，如多尺度分解、稀疏表示、深度学习', 'varchar(100)', 'String', 'name', '0', '0', '1', '1', '1', '1', '1', 'LIKE', 'input', '', 2, 'admin', '2025-05-20 16:43:14', '', '2025-05-21 18:01:58');
INSERT INTO `gen_table_column` VALUES (13, 4, 'id', NULL, 'int', 'Long', 'id', '1', '1', '0', '0', NULL, NULL, NULL, 'EQ', 'input', '', 1, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (14, 4, 'name', '视频标题', 'varchar(200)', 'String', 'name', '0', '0', '1', '1', '1', '1', '1', 'LIKE', 'input', '', 2, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (15, 4, 'storage_path', '文件存储路径', 'varchar(500)', 'String', 'storagePath', '0', '0', '1', '1', '1', '1', '1', 'EQ', 'textarea', '', 3, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (16, 4, 'duration', '视频时长', 'time', 'Date', 'duration', '0', '0', '1', '1', '1', '1', '1', 'EQ', 'datetime', '', 4, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (17, 4, 'description', '视频简介', 'text', 'String', 'description', '0', '0', '0', '1', '1', '1', '1', 'EQ', 'textarea', '', 5, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (18, 4, 'audience', '适用人群', 'varchar(100)', 'String', 'audience', '0', '0', '0', '1', '1', '1', '1', 'EQ', 'input', '', 6, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (19, 4, 'cover_path', '封面图像路径', 'varchar(500)', 'String', 'coverPath', '0', '0', '0', '1', '1', '1', '1', 'EQ', 'textarea', '', 7, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (20, 4, 'category_id', '关联 video_category', 'int', 'Long', 'categoryId', '0', '0', '1', '1', '1', '1', '1', 'EQ', 'input', '', 8, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (21, 4, 'created_at', NULL, 'datetime', 'Date', 'createdAt', '0', '0', '0', '0', '0', '0', '0', 'EQ', 'datetime', '', 9, 'admin', '2025-05-20 16:43:14', '', '2025-05-22 12:32:43');
INSERT INTO `gen_table_column` VALUES (22, 2, 'cover_path', '封面图像路径', 'varchar(500)', 'String', 'coverPath', '0', '0', '0', '1', '1', '1', '1', 'EQ', 'textarea', '', 7, '', '2025-05-25 02:52:26', '', '2025-05-25 02:52:51');

-- ----------------------------
-- Table structure for qrtz_blob_triggers
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_blob_triggers`;
CREATE TABLE `qrtz_blob_triggers`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `trigger_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_name的外键',
  `trigger_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_group的外键',
  `blob_data` blob NULL COMMENT '存放持久化Trigger对象',
  PRIMARY KEY (`sched_name`, `trigger_name`, `trigger_group`) USING BTREE,
  CONSTRAINT `qrtz_blob_triggers_ibfk_1` FOREIGN KEY (`sched_name`, `trigger_name`, `trigger_group`) REFERENCES `qrtz_triggers` (`sched_name`, `trigger_name`, `trigger_group`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = 'Blob类型的触发器表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_blob_triggers
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_calendars
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_calendars`;
CREATE TABLE `qrtz_calendars`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `calendar_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '日历名称',
  `calendar` blob NOT NULL COMMENT '存放持久化calendar对象',
  PRIMARY KEY (`sched_name`, `calendar_name`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '日历信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_calendars
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_cron_triggers
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_cron_triggers`;
CREATE TABLE `qrtz_cron_triggers`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `trigger_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_name的外键',
  `trigger_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_group的外键',
  `cron_expression` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'cron表达式',
  `time_zone_id` varchar(80) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '时区',
  PRIMARY KEY (`sched_name`, `trigger_name`, `trigger_group`) USING BTREE,
  CONSTRAINT `qrtz_cron_triggers_ibfk_1` FOREIGN KEY (`sched_name`, `trigger_name`, `trigger_group`) REFERENCES `qrtz_triggers` (`sched_name`, `trigger_name`, `trigger_group`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = 'Cron类型的触发器表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_cron_triggers
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_fired_triggers
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_fired_triggers`;
CREATE TABLE `qrtz_fired_triggers`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `entry_id` varchar(95) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度器实例id',
  `trigger_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_name的外键',
  `trigger_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_group的外键',
  `instance_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度器实例名',
  `fired_time` bigint NOT NULL COMMENT '触发的时间',
  `sched_time` bigint NOT NULL COMMENT '定时器制定的时间',
  `priority` int NOT NULL COMMENT '优先级',
  `state` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '状态',
  `job_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '任务名称',
  `job_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '任务组名',
  `is_nonconcurrent` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否并发',
  `requests_recovery` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '是否接受恢复执行',
  PRIMARY KEY (`sched_name`, `entry_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '已触发的触发器表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_fired_triggers
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_job_details
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_job_details`;
CREATE TABLE `qrtz_job_details`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `job_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '任务名称',
  `job_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '任务组名',
  `description` varchar(250) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '相关介绍',
  `job_class_name` varchar(250) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '执行任务类名称',
  `is_durable` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '是否持久化',
  `is_nonconcurrent` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '是否并发',
  `is_update_data` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '是否更新数据',
  `requests_recovery` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '是否接受恢复执行',
  `job_data` blob NULL COMMENT '存放持久化job对象',
  PRIMARY KEY (`sched_name`, `job_name`, `job_group`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '任务详细信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_job_details
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_locks
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_locks`;
CREATE TABLE `qrtz_locks`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `lock_name` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '悲观锁名称',
  PRIMARY KEY (`sched_name`, `lock_name`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '存储的悲观锁信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_locks
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_paused_trigger_grps
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_paused_trigger_grps`;
CREATE TABLE `qrtz_paused_trigger_grps`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `trigger_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_group的外键',
  PRIMARY KEY (`sched_name`, `trigger_group`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '暂停的触发器表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_paused_trigger_grps
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_scheduler_state
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_scheduler_state`;
CREATE TABLE `qrtz_scheduler_state`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `instance_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '实例名称',
  `last_checkin_time` bigint NOT NULL COMMENT '上次检查时间',
  `checkin_interval` bigint NOT NULL COMMENT '检查间隔时间',
  PRIMARY KEY (`sched_name`, `instance_name`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '调度器状态表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_scheduler_state
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_simple_triggers
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_simple_triggers`;
CREATE TABLE `qrtz_simple_triggers`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `trigger_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_name的外键',
  `trigger_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_group的外键',
  `repeat_count` bigint NOT NULL COMMENT '重复的次数统计',
  `repeat_interval` bigint NOT NULL COMMENT '重复的间隔时间',
  `times_triggered` bigint NOT NULL COMMENT '已经触发的次数',
  PRIMARY KEY (`sched_name`, `trigger_name`, `trigger_group`) USING BTREE,
  CONSTRAINT `qrtz_simple_triggers_ibfk_1` FOREIGN KEY (`sched_name`, `trigger_name`, `trigger_group`) REFERENCES `qrtz_triggers` (`sched_name`, `trigger_name`, `trigger_group`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '简单触发器的信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_simple_triggers
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_simprop_triggers
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_simprop_triggers`;
CREATE TABLE `qrtz_simprop_triggers`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `trigger_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_name的外键',
  `trigger_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_triggers表trigger_group的外键',
  `str_prop_1` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT 'String类型的trigger的第一个参数',
  `str_prop_2` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT 'String类型的trigger的第二个参数',
  `str_prop_3` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT 'String类型的trigger的第三个参数',
  `int_prop_1` int NULL DEFAULT NULL COMMENT 'int类型的trigger的第一个参数',
  `int_prop_2` int NULL DEFAULT NULL COMMENT 'int类型的trigger的第二个参数',
  `long_prop_1` bigint NULL DEFAULT NULL COMMENT 'long类型的trigger的第一个参数',
  `long_prop_2` bigint NULL DEFAULT NULL COMMENT 'long类型的trigger的第二个参数',
  `dec_prop_1` decimal(13, 4) NULL DEFAULT NULL COMMENT 'decimal类型的trigger的第一个参数',
  `dec_prop_2` decimal(13, 4) NULL DEFAULT NULL COMMENT 'decimal类型的trigger的第二个参数',
  `bool_prop_1` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT 'Boolean类型的trigger的第一个参数',
  `bool_prop_2` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT 'Boolean类型的trigger的第二个参数',
  PRIMARY KEY (`sched_name`, `trigger_name`, `trigger_group`) USING BTREE,
  CONSTRAINT `qrtz_simprop_triggers_ibfk_1` FOREIGN KEY (`sched_name`, `trigger_name`, `trigger_group`) REFERENCES `qrtz_triggers` (`sched_name`, `trigger_name`, `trigger_group`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '同步机制的行锁表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_simprop_triggers
-- ----------------------------

-- ----------------------------
-- Table structure for qrtz_triggers
-- ----------------------------
DROP TABLE IF EXISTS `qrtz_triggers`;
CREATE TABLE `qrtz_triggers`  (
  `sched_name` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调度名称',
  `trigger_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '触发器的名字',
  `trigger_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '触发器所属组的名字',
  `job_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_job_details表job_name的外键',
  `job_group` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT 'qrtz_job_details表job_group的外键',
  `description` varchar(250) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '相关介绍',
  `next_fire_time` bigint NULL DEFAULT NULL COMMENT '上一次触发时间（毫秒）',
  `prev_fire_time` bigint NULL DEFAULT NULL COMMENT '下一次触发时间（默认为-1表示不触发）',
  `priority` int NULL DEFAULT NULL COMMENT '优先级',
  `trigger_state` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '触发器状态',
  `trigger_type` varchar(8) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '触发器的类型',
  `start_time` bigint NOT NULL COMMENT '开始时间',
  `end_time` bigint NULL DEFAULT NULL COMMENT '结束时间',
  `calendar_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '日程表名称',
  `misfire_instr` smallint NULL DEFAULT NULL COMMENT '补偿执行的策略',
  `job_data` blob NULL COMMENT '存放持久化job对象',
  PRIMARY KEY (`sched_name`, `trigger_name`, `trigger_group`) USING BTREE,
  INDEX `sched_name`(`sched_name` ASC, `job_name` ASC, `job_group` ASC) USING BTREE,
  CONSTRAINT `qrtz_triggers_ibfk_1` FOREIGN KEY (`sched_name`, `job_name`, `job_group`) REFERENCES `qrtz_job_details` (`sched_name`, `job_name`, `job_group`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '触发器详细信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of qrtz_triggers
-- ----------------------------

-- ----------------------------
-- Table structure for sys_config
-- ----------------------------
DROP TABLE IF EXISTS `sys_config`;
CREATE TABLE `sys_config`  (
  `config_id` int NOT NULL AUTO_INCREMENT COMMENT '参数主键',
  `config_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '参数名称',
  `config_key` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '参数键名',
  `config_value` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '参数键值',
  `config_type` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT 'N' COMMENT '系统内置（Y是 N否）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`config_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 100 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '参数配置表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_config
-- ----------------------------
INSERT INTO `sys_config` VALUES (1, '主框架页-默认皮肤样式名称', 'sys.index.skinName', 'skin-blue', 'Y', 'admin', '2025-05-17 15:01:03', '', NULL, '蓝色 skin-blue、绿色 skin-green、紫色 skin-purple、红色 skin-red、黄色 skin-yellow');
INSERT INTO `sys_config` VALUES (2, '用户管理-账号初始密码', 'sys.user.initPassword', '123456', 'Y', 'admin', '2025-05-17 15:01:03', '', NULL, '初始化密码 123456');
INSERT INTO `sys_config` VALUES (3, '主框架页-侧边栏主题', 'sys.index.sideTheme', 'theme-dark', 'Y', 'admin', '2025-05-17 15:01:03', '', NULL, '深色主题theme-dark，浅色主题theme-light');
INSERT INTO `sys_config` VALUES (4, '账号自助-验证码开关', 'sys.account.captchaEnabled', 'true', 'Y', 'admin', '2025-05-17 15:01:03', '', NULL, '是否开启验证码功能（true开启，false关闭）');
INSERT INTO `sys_config` VALUES (5, '账号自助-是否开启用户注册功能', 'sys.account.registerUser', 'false', 'Y', 'admin', '2025-05-17 15:01:03', '', NULL, '是否开启注册用户功能（true开启，false关闭）');
INSERT INTO `sys_config` VALUES (6, '用户登录-黑名单列表', 'sys.login.blackIPList', '', 'Y', 'admin', '2025-05-17 15:01:03', '', NULL, '设置登录IP黑名单限制，多个匹配项以;分隔，支持匹配（*通配、网段）');

-- ----------------------------
-- Table structure for sys_dept
-- ----------------------------
DROP TABLE IF EXISTS `sys_dept`;
CREATE TABLE `sys_dept`  (
  `dept_id` bigint NOT NULL AUTO_INCREMENT COMMENT '部门id',
  `parent_id` bigint NULL DEFAULT 0 COMMENT '父部门id',
  `ancestors` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '祖级列表',
  `dept_name` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '部门名称',
  `order_num` int NULL DEFAULT 0 COMMENT '显示顺序',
  `leader` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '负责人',
  `phone` varchar(11) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '联系电话',
  `email` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '邮箱',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '部门状态（0正常 1停用）',
  `del_flag` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '删除标志（0代表存在 2代表删除）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`dept_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 200 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '部门表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_dept
-- ----------------------------
INSERT INTO `sys_dept` VALUES (100, 0, '0', '若依科技', 0, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (101, 100, '0,100', '深圳总公司', 1, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (102, 100, '0,100', '长沙分公司', 2, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (103, 101, '0,100,101', '研发部门', 1, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (104, 101, '0,100,101', '市场部门', 2, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (105, 101, '0,100,101', '测试部门', 3, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (106, 101, '0,100,101', '财务部门', 4, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (107, 101, '0,100,101', '运维部门', 5, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (108, 102, '0,100,102', '市场部门', 1, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);
INSERT INTO `sys_dept` VALUES (109, 102, '0,100,102', '财务部门', 2, '若依', '15888888888', 'ry@qq.com', '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL);

-- ----------------------------
-- Table structure for sys_dict_data
-- ----------------------------
DROP TABLE IF EXISTS `sys_dict_data`;
CREATE TABLE `sys_dict_data`  (
  `dict_code` bigint NOT NULL AUTO_INCREMENT COMMENT '字典编码',
  `dict_sort` int NULL DEFAULT 0 COMMENT '字典排序',
  `dict_label` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '字典标签',
  `dict_value` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '字典键值',
  `dict_type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '字典类型',
  `css_class` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '样式属性（其他样式扩展）',
  `list_class` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '表格回显样式',
  `is_default` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT 'N' COMMENT '是否默认（Y是 N否）',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '状态（0正常 1停用）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`dict_code`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 100 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '字典数据表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_dict_data
-- ----------------------------
INSERT INTO `sys_dict_data` VALUES (1, 1, '男', '0', 'sys_user_sex', '', '', 'Y', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '性别男');
INSERT INTO `sys_dict_data` VALUES (2, 2, '女', '1', 'sys_user_sex', '', '', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '性别女');
INSERT INTO `sys_dict_data` VALUES (3, 3, '未知', '2', 'sys_user_sex', '', '', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '性别未知');
INSERT INTO `sys_dict_data` VALUES (4, 1, '显示', '0', 'sys_show_hide', '', 'primary', 'Y', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '显示菜单');
INSERT INTO `sys_dict_data` VALUES (5, 2, '隐藏', '1', 'sys_show_hide', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '隐藏菜单');
INSERT INTO `sys_dict_data` VALUES (6, 1, '正常', '0', 'sys_normal_disable', '', 'primary', 'Y', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '正常状态');
INSERT INTO `sys_dict_data` VALUES (7, 2, '停用', '1', 'sys_normal_disable', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '停用状态');
INSERT INTO `sys_dict_data` VALUES (8, 1, '正常', '0', 'sys_job_status', '', 'primary', 'Y', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '正常状态');
INSERT INTO `sys_dict_data` VALUES (9, 2, '暂停', '1', 'sys_job_status', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '停用状态');
INSERT INTO `sys_dict_data` VALUES (10, 1, '默认', 'DEFAULT', 'sys_job_group', '', '', 'Y', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '默认分组');
INSERT INTO `sys_dict_data` VALUES (11, 2, '系统', 'SYSTEM', 'sys_job_group', '', '', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '系统分组');
INSERT INTO `sys_dict_data` VALUES (12, 1, '是', 'Y', 'sys_yes_no', '', 'primary', 'Y', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '系统默认是');
INSERT INTO `sys_dict_data` VALUES (13, 2, '否', 'N', 'sys_yes_no', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '系统默认否');
INSERT INTO `sys_dict_data` VALUES (14, 1, '通知', '1', 'sys_notice_type', '', 'warning', 'Y', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '通知');
INSERT INTO `sys_dict_data` VALUES (15, 2, '公告', '2', 'sys_notice_type', '', 'success', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '公告');
INSERT INTO `sys_dict_data` VALUES (16, 1, '正常', '0', 'sys_notice_status', '', 'primary', 'Y', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '正常状态');
INSERT INTO `sys_dict_data` VALUES (17, 2, '关闭', '1', 'sys_notice_status', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '关闭状态');
INSERT INTO `sys_dict_data` VALUES (18, 99, '其他', '0', 'sys_oper_type', '', 'info', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '其他操作');
INSERT INTO `sys_dict_data` VALUES (19, 1, '新增', '1', 'sys_oper_type', '', 'info', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '新增操作');
INSERT INTO `sys_dict_data` VALUES (20, 2, '修改', '2', 'sys_oper_type', '', 'info', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '修改操作');
INSERT INTO `sys_dict_data` VALUES (21, 3, '删除', '3', 'sys_oper_type', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '删除操作');
INSERT INTO `sys_dict_data` VALUES (22, 4, '授权', '4', 'sys_oper_type', '', 'primary', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '授权操作');
INSERT INTO `sys_dict_data` VALUES (23, 5, '导出', '5', 'sys_oper_type', '', 'warning', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '导出操作');
INSERT INTO `sys_dict_data` VALUES (24, 6, '导入', '6', 'sys_oper_type', '', 'warning', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '导入操作');
INSERT INTO `sys_dict_data` VALUES (25, 7, '强退', '7', 'sys_oper_type', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '强退操作');
INSERT INTO `sys_dict_data` VALUES (26, 8, '生成代码', '8', 'sys_oper_type', '', 'warning', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '生成操作');
INSERT INTO `sys_dict_data` VALUES (27, 9, '清空数据', '9', 'sys_oper_type', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '清空操作');
INSERT INTO `sys_dict_data` VALUES (28, 1, '成功', '0', 'sys_common_status', '', 'primary', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '正常状态');
INSERT INTO `sys_dict_data` VALUES (29, 2, '失败', '1', 'sys_common_status', '', 'danger', 'N', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '停用状态');

-- ----------------------------
-- Table structure for sys_dict_type
-- ----------------------------
DROP TABLE IF EXISTS `sys_dict_type`;
CREATE TABLE `sys_dict_type`  (
  `dict_id` bigint NOT NULL AUTO_INCREMENT COMMENT '字典主键',
  `dict_name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '字典名称',
  `dict_type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '字典类型',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '状态（0正常 1停用）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`dict_id`) USING BTREE,
  UNIQUE INDEX `dict_type`(`dict_type` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 100 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '字典类型表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_dict_type
-- ----------------------------
INSERT INTO `sys_dict_type` VALUES (1, '用户性别', 'sys_user_sex', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '用户性别列表');
INSERT INTO `sys_dict_type` VALUES (2, '菜单状态', 'sys_show_hide', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '菜单状态列表');
INSERT INTO `sys_dict_type` VALUES (3, '系统开关', 'sys_normal_disable', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '系统开关列表');
INSERT INTO `sys_dict_type` VALUES (4, '任务状态', 'sys_job_status', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '任务状态列表');
INSERT INTO `sys_dict_type` VALUES (5, '任务分组', 'sys_job_group', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '任务分组列表');
INSERT INTO `sys_dict_type` VALUES (6, '系统是否', 'sys_yes_no', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '系统是否列表');
INSERT INTO `sys_dict_type` VALUES (7, '通知类型', 'sys_notice_type', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '通知类型列表');
INSERT INTO `sys_dict_type` VALUES (8, '通知状态', 'sys_notice_status', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '通知状态列表');
INSERT INTO `sys_dict_type` VALUES (9, '操作类型', 'sys_oper_type', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '操作类型列表');
INSERT INTO `sys_dict_type` VALUES (10, '系统状态', 'sys_common_status', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '登录状态列表');

-- ----------------------------
-- Table structure for sys_job
-- ----------------------------
DROP TABLE IF EXISTS `sys_job`;
CREATE TABLE `sys_job`  (
  `job_id` bigint NOT NULL AUTO_INCREMENT COMMENT '任务ID',
  `job_name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL DEFAULT '' COMMENT '任务名称',
  `job_group` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL DEFAULT 'DEFAULT' COMMENT '任务组名',
  `invoke_target` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调用目标字符串',
  `cron_expression` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT 'cron执行表达式',
  `misfire_policy` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '3' COMMENT '计划执行错误策略（1立即执行 2执行一次 3放弃执行）',
  `concurrent` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '1' COMMENT '是否并发执行（0允许 1禁止）',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '状态（0正常 1暂停）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '备注信息',
  PRIMARY KEY (`job_id`, `job_name`, `job_group`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 100 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '定时任务调度表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_job
-- ----------------------------
INSERT INTO `sys_job` VALUES (1, '系统默认（无参）', 'DEFAULT', 'ryTask.ryNoParams', '0/10 * * * * ?', '3', '1', '1', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_job` VALUES (2, '系统默认（有参）', 'DEFAULT', 'ryTask.ryParams(\'ry\')', '0/15 * * * * ?', '3', '1', '1', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_job` VALUES (3, '系统默认（多参）', 'DEFAULT', 'ryTask.ryMultipleParams(\'ry\', true, 2000L, 316.50D, 100)', '0/20 * * * * ?', '3', '1', '1', 'admin', '2025-05-17 15:01:03', '', NULL, '');

-- ----------------------------
-- Table structure for sys_job_log
-- ----------------------------
DROP TABLE IF EXISTS `sys_job_log`;
CREATE TABLE `sys_job_log`  (
  `job_log_id` bigint NOT NULL AUTO_INCREMENT COMMENT '任务日志ID',
  `job_name` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '任务名称',
  `job_group` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '任务组名',
  `invoke_target` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '调用目标字符串',
  `job_message` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '日志信息',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '执行状态（0正常 1失败）',
  `exception_info` varchar(2000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '异常信息',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  PRIMARY KEY (`job_log_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '定时任务调度日志表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_job_log
-- ----------------------------

-- ----------------------------
-- Table structure for sys_logininfor
-- ----------------------------
DROP TABLE IF EXISTS `sys_logininfor`;
CREATE TABLE `sys_logininfor`  (
  `info_id` bigint NOT NULL AUTO_INCREMENT COMMENT '访问ID',
  `user_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '用户账号',
  `ipaddr` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '登录IP地址',
  `login_location` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '登录地点',
  `browser` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '浏览器类型',
  `os` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '操作系统',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '登录状态（0成功 1失败）',
  `msg` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '提示消息',
  `login_time` datetime NULL DEFAULT NULL COMMENT '访问时间',
  PRIMARY KEY (`info_id`) USING BTREE,
  INDEX `idx_sys_logininfor_s`(`status` ASC) USING BTREE,
  INDEX `idx_sys_logininfor_lt`(`login_time` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 182 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '系统访问记录' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_logininfor
-- ----------------------------
INSERT INTO `sys_logininfor` VALUES (100, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-17 23:10:36');
INSERT INTO `sys_logininfor` VALUES (101, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-17 23:10:40');
INSERT INTO `sys_logininfor` VALUES (102, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-17 23:10:46');
INSERT INTO `sys_logininfor` VALUES (103, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-18 16:32:19');
INSERT INTO `sys_logininfor` VALUES (104, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-20 10:23:08');
INSERT INTO `sys_logininfor` VALUES (105, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-20 14:30:59');
INSERT INTO `sys_logininfor` VALUES (106, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-20 14:31:07');
INSERT INTO `sys_logininfor` VALUES (107, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-20 14:49:32');
INSERT INTO `sys_logininfor` VALUES (108, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-20 14:51:03');
INSERT INTO `sys_logininfor` VALUES (109, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-20 14:56:43');
INSERT INTO `sys_logininfor` VALUES (110, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-20 15:01:11');
INSERT INTO `sys_logininfor` VALUES (111, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-20 16:41:17');
INSERT INTO `sys_logininfor` VALUES (112, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-21 12:54:18');
INSERT INTO `sys_logininfor` VALUES (113, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码已失效', '2025-05-21 17:45:25');
INSERT INTO `sys_logininfor` VALUES (114, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-21 17:45:29');
INSERT INTO `sys_logininfor` VALUES (115, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-21 20:07:48');
INSERT INTO `sys_logininfor` VALUES (116, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-21 21:39:09');
INSERT INTO `sys_logininfor` VALUES (117, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-22 10:22:58');
INSERT INTO `sys_logininfor` VALUES (118, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-22 10:48:10');
INSERT INTO `sys_logininfor` VALUES (119, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-22 12:23:54');
INSERT INTO `sys_logininfor` VALUES (120, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-22 12:23:56');
INSERT INTO `sys_logininfor` VALUES (121, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-22 15:52:38');
INSERT INTO `sys_logininfor` VALUES (122, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码已失效', '2025-05-22 16:32:40');
INSERT INTO `sys_logininfor` VALUES (123, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-22 16:32:42');
INSERT INTO `sys_logininfor` VALUES (124, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-22 16:32:48');
INSERT INTO `sys_logininfor` VALUES (125, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-22 18:16:49');
INSERT INTO `sys_logininfor` VALUES (126, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-22 18:16:53');
INSERT INTO `sys_logininfor` VALUES (127, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-22 19:12:08');
INSERT INTO `sys_logininfor` VALUES (128, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-22 22:05:32');
INSERT INTO `sys_logininfor` VALUES (129, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-23 00:50:11');
INSERT INTO `sys_logininfor` VALUES (130, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-23 08:34:06');
INSERT INTO `sys_logininfor` VALUES (131, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-23 12:56:49');
INSERT INTO `sys_logininfor` VALUES (132, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-23 18:31:33');
INSERT INTO `sys_logininfor` VALUES (133, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-23 21:50:02');
INSERT INTO `sys_logininfor` VALUES (134, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码已失效', '2025-05-24 02:54:02');
INSERT INTO `sys_logininfor` VALUES (135, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 02:54:07');
INSERT INTO `sys_logininfor` VALUES (136, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 08:21:01');
INSERT INTO `sys_logininfor` VALUES (137, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 10:34:51');
INSERT INTO `sys_logininfor` VALUES (138, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-24 12:07:48');
INSERT INTO `sys_logininfor` VALUES (139, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 12:07:52');
INSERT INTO `sys_logininfor` VALUES (140, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码已失效', '2025-05-24 12:53:52');
INSERT INTO `sys_logininfor` VALUES (141, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 12:54:06');
INSERT INTO `sys_logininfor` VALUES (142, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 12:54:30');
INSERT INTO `sys_logininfor` VALUES (143, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 13:41:16');
INSERT INTO `sys_logininfor` VALUES (144, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 13:49:08');
INSERT INTO `sys_logininfor` VALUES (145, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 15:02:06');
INSERT INTO `sys_logininfor` VALUES (146, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-24 17:02:30');
INSERT INTO `sys_logininfor` VALUES (147, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 17:02:38');
INSERT INTO `sys_logininfor` VALUES (148, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码已失效', '2025-05-24 17:14:21');
INSERT INTO `sys_logininfor` VALUES (149, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 17:14:23');
INSERT INTO `sys_logininfor` VALUES (150, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码已失效', '2025-05-24 18:51:34');
INSERT INTO `sys_logininfor` VALUES (151, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 18:51:51');
INSERT INTO `sys_logininfor` VALUES (152, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-24 18:53:33');
INSERT INTO `sys_logininfor` VALUES (153, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 18:53:36');
INSERT INTO `sys_logininfor` VALUES (154, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-24 18:58:07');
INSERT INTO `sys_logininfor` VALUES (155, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 18:58:27');
INSERT INTO `sys_logininfor` VALUES (156, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 21:10:25');
INSERT INTO `sys_logininfor` VALUES (157, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-24 21:28:52');
INSERT INTO `sys_logininfor` VALUES (158, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 21:28:59');
INSERT INTO `sys_logininfor` VALUES (159, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-24 21:29:03');
INSERT INTO `sys_logininfor` VALUES (160, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 21:29:10');
INSERT INTO `sys_logininfor` VALUES (161, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 22:42:42');
INSERT INTO `sys_logininfor` VALUES (162, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-24 23:45:33');
INSERT INTO `sys_logininfor` VALUES (163, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-25 01:55:47');
INSERT INTO `sys_logininfor` VALUES (164, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 01:55:51');
INSERT INTO `sys_logininfor` VALUES (165, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 09:39:09');
INSERT INTO `sys_logininfor` VALUES (166, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-25 10:36:34');
INSERT INTO `sys_logininfor` VALUES (167, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 10:36:37');
INSERT INTO `sys_logininfor` VALUES (168, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 11:42:49');
INSERT INTO `sys_logininfor` VALUES (169, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 12:17:17');
INSERT INTO `sys_logininfor` VALUES (170, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 12:58:42');
INSERT INTO `sys_logininfor` VALUES (171, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-25 14:25:51');
INSERT INTO `sys_logininfor` VALUES (172, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 14:25:54');
INSERT INTO `sys_logininfor` VALUES (173, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-25 15:45:23');
INSERT INTO `sys_logininfor` VALUES (174, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 15:45:26');
INSERT INTO `sys_logininfor` VALUES (175, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 16:16:55');
INSERT INTO `sys_logininfor` VALUES (176, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 16:17:06');
INSERT INTO `sys_logininfor` VALUES (177, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-25 16:21:53');
INSERT INTO `sys_logininfor` VALUES (178, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '1', '验证码错误', '2025-05-25 16:22:01');
INSERT INTO `sys_logininfor` VALUES (179, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '登录成功', '2025-05-25 16:22:03');
INSERT INTO `sys_logininfor` VALUES (180, 'ry', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-25 17:01:41');
INSERT INTO `sys_logininfor` VALUES (181, 'admin', '127.0.0.1', '内网IP', 'Chrome 13', 'Windows 10', '0', '退出成功', '2025-05-25 17:01:58');

-- ----------------------------
-- Table structure for sys_menu
-- ----------------------------
DROP TABLE IF EXISTS `sys_menu`;
CREATE TABLE `sys_menu`  (
  `menu_id` bigint NOT NULL AUTO_INCREMENT COMMENT '菜单ID',
  `menu_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '菜单名称',
  `parent_id` bigint NULL DEFAULT 0 COMMENT '父菜单ID',
  `order_num` int NULL DEFAULT 0 COMMENT '显示顺序',
  `path` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '路由地址',
  `component` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '组件路径',
  `query` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '路由参数',
  `route_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '路由名称',
  `is_frame` int NULL DEFAULT 1 COMMENT '是否为外链（0是 1否）',
  `is_cache` int NULL DEFAULT 0 COMMENT '是否缓存（0缓存 1不缓存）',
  `menu_type` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '菜单类型（M目录 C菜单 F按钮）',
  `visible` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '菜单状态（0显示 1隐藏）',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '菜单状态（0正常 1停用）',
  `perms` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '权限标识',
  `icon` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '#' COMMENT '菜单图标',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '备注',
  PRIMARY KEY (`menu_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2085 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '菜单权限表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_menu
-- ----------------------------
INSERT INTO `sys_menu` VALUES (1, '系统管理', 0, 1, 'system', NULL, '', '', 1, 0, 'M', '0', '0', '', 'system', 'admin', '2025-05-17 15:01:03', '', NULL, '系统管理目录');
INSERT INTO `sys_menu` VALUES (2, '系统监控', 0, 2, 'monitor', NULL, '', '', 1, 0, 'M', '0', '0', '', 'monitor', 'admin', '2025-05-17 15:01:03', '', NULL, '系统监控目录');
INSERT INTO `sys_menu` VALUES (3, '系统工具', 0, 3, 'tool', NULL, '', '', 1, 0, 'M', '0', '0', '', 'tool', 'admin', '2025-05-17 15:01:03', '', NULL, '系统工具目录');
INSERT INTO `sys_menu` VALUES (4, '若依官网', 0, 4, 'http://ruoyi.vip', NULL, '', '', 0, 0, 'M', '1', '1', '', 'guide', 'admin', '2025-05-17 15:01:03', 'ry', '2025-05-24 18:54:44', '若依官网地址');
INSERT INTO `sys_menu` VALUES (100, '用户管理', 1, 1, 'user', 'system/user/index', '', '', 1, 0, 'C', '0', '0', 'system:user:list', 'user', 'admin', '2025-05-17 15:01:03', '', NULL, '用户管理菜单');
INSERT INTO `sys_menu` VALUES (101, '角色管理', 1, 2, 'role', 'system/role/index', '', '', 1, 0, 'C', '0', '0', 'system:role:list', 'peoples', 'admin', '2025-05-17 15:01:03', '', NULL, '角色管理菜单');
INSERT INTO `sys_menu` VALUES (102, '菜单管理', 1, 3, 'menu', 'system/menu/index', '', '', 1, 0, 'C', '0', '0', 'system:menu:list', 'tree-table', 'admin', '2025-05-17 15:01:03', '', NULL, '菜单管理菜单');
INSERT INTO `sys_menu` VALUES (103, '部门管理', 1, 4, 'dept', 'system/dept/index', '', '', 1, 0, 'C', '1', '1', 'system:dept:list', 'tree', 'admin', '2025-05-17 15:01:03', 'ry', '2025-05-24 18:56:05', '部门管理菜单');
INSERT INTO `sys_menu` VALUES (104, '岗位管理', 1, 5, 'post', 'system/post/index', '', '', 1, 0, 'C', '1', '1', 'system:post:list', 'post', 'admin', '2025-05-17 15:01:03', 'ry', '2025-05-24 18:56:10', '岗位管理菜单');
INSERT INTO `sys_menu` VALUES (105, '字典管理', 1, 6, 'dict', 'system/dict/index', '', '', 1, 0, 'C', '0', '0', 'system:dict:list', 'dict', 'admin', '2025-05-17 15:01:03', '', NULL, '字典管理菜单');
INSERT INTO `sys_menu` VALUES (106, '参数设置', 1, 7, 'config', 'system/config/index', '', '', 1, 0, 'C', '0', '0', 'system:config:list', 'edit', 'admin', '2025-05-17 15:01:03', '', NULL, '参数设置菜单');
INSERT INTO `sys_menu` VALUES (107, '通知公告', 1, 8, 'notice', 'system/notice/index', '', '', 1, 0, 'C', '0', '0', 'system:notice:list', 'message', 'admin', '2025-05-17 15:01:03', '', NULL, '通知公告菜单');
INSERT INTO `sys_menu` VALUES (108, '日志管理', 1, 9, 'log', '', '', '', 1, 0, 'M', '0', '0', '', 'log', 'admin', '2025-05-17 15:01:03', '', NULL, '日志管理菜单');
INSERT INTO `sys_menu` VALUES (109, '在线用户', 2, 1, 'online', 'monitor/online/index', '', '', 1, 0, 'C', '0', '0', 'monitor:online:list', 'online', 'admin', '2025-05-17 15:01:03', '', NULL, '在线用户菜单');
INSERT INTO `sys_menu` VALUES (110, '定时任务', 2, 2, 'job', 'monitor/job/index', '', '', 1, 0, 'C', '0', '0', 'monitor:job:list', 'job', 'admin', '2025-05-17 15:01:03', '', NULL, '定时任务菜单');
INSERT INTO `sys_menu` VALUES (111, '数据监控', 2, 3, 'druid', 'monitor/druid/index', '', '', 1, 0, 'C', '0', '0', 'monitor:druid:list', 'druid', 'admin', '2025-05-17 15:01:03', '', NULL, '数据监控菜单');
INSERT INTO `sys_menu` VALUES (112, '服务监控', 2, 4, 'server', 'monitor/server/index', '', '', 1, 0, 'C', '0', '0', 'monitor:server:list', 'server', 'admin', '2025-05-17 15:01:03', '', NULL, '服务监控菜单');
INSERT INTO `sys_menu` VALUES (113, '缓存监控', 2, 5, 'cache', 'monitor/cache/index', '', '', 1, 0, 'C', '0', '0', 'monitor:cache:list', 'redis', 'admin', '2025-05-17 15:01:03', '', NULL, '缓存监控菜单');
INSERT INTO `sys_menu` VALUES (114, '缓存列表', 2, 6, 'cacheList', 'monitor/cache/list', '', '', 1, 0, 'C', '0', '0', 'monitor:cache:list', 'redis-list', 'admin', '2025-05-17 15:01:03', '', NULL, '缓存列表菜单');
INSERT INTO `sys_menu` VALUES (115, '表单构建', 3, 1, 'build', 'tool/build/index', '', '', 1, 0, 'C', '0', '0', 'tool:build:list', 'build', 'admin', '2025-05-17 15:01:03', '', NULL, '表单构建菜单');
INSERT INTO `sys_menu` VALUES (116, '代码生成', 3, 2, 'gen', 'tool/gen/index', '', '', 1, 0, 'C', '0', '0', 'tool:gen:list', 'code', 'admin', '2025-05-17 15:01:03', '', NULL, '代码生成菜单');
INSERT INTO `sys_menu` VALUES (117, '系统接口', 3, 3, 'swagger', 'tool/swagger/index', '', '', 1, 0, 'C', '0', '0', 'tool:swagger:list', 'swagger', 'admin', '2025-05-17 15:01:03', '', NULL, '系统接口菜单');
INSERT INTO `sys_menu` VALUES (500, '操作日志', 108, 1, 'operlog', 'monitor/operlog/index', '', '', 1, 0, 'C', '0', '0', 'monitor:operlog:list', 'form', 'admin', '2025-05-17 15:01:03', '', NULL, '操作日志菜单');
INSERT INTO `sys_menu` VALUES (501, '登录日志', 108, 2, 'logininfor', 'monitor/logininfor/index', '', '', 1, 0, 'C', '0', '0', 'monitor:logininfor:list', 'logininfor', 'admin', '2025-05-17 15:01:03', '', NULL, '登录日志菜单');
INSERT INTO `sys_menu` VALUES (1000, '用户查询', 100, 1, '', '', '', '', 1, 0, 'F', '0', '0', 'system:user:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1001, '用户新增', 100, 2, '', '', '', '', 1, 0, 'F', '0', '0', 'system:user:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1002, '用户修改', 100, 3, '', '', '', '', 1, 0, 'F', '0', '0', 'system:user:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1003, '用户删除', 100, 4, '', '', '', '', 1, 0, 'F', '0', '0', 'system:user:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1004, '用户导出', 100, 5, '', '', '', '', 1, 0, 'F', '0', '0', 'system:user:export', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1005, '用户导入', 100, 6, '', '', '', '', 1, 0, 'F', '0', '0', 'system:user:import', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1006, '重置密码', 100, 7, '', '', '', '', 1, 0, 'F', '0', '0', 'system:user:resetPwd', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1007, '角色查询', 101, 1, '', '', '', '', 1, 0, 'F', '0', '0', 'system:role:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1008, '角色新增', 101, 2, '', '', '', '', 1, 0, 'F', '0', '0', 'system:role:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1009, '角色修改', 101, 3, '', '', '', '', 1, 0, 'F', '0', '0', 'system:role:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1010, '角色删除', 101, 4, '', '', '', '', 1, 0, 'F', '0', '0', 'system:role:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1011, '角色导出', 101, 5, '', '', '', '', 1, 0, 'F', '0', '0', 'system:role:export', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1012, '菜单查询', 102, 1, '', '', '', '', 1, 0, 'F', '0', '0', 'system:menu:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1013, '菜单新增', 102, 2, '', '', '', '', 1, 0, 'F', '0', '0', 'system:menu:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1014, '菜单修改', 102, 3, '', '', '', '', 1, 0, 'F', '0', '0', 'system:menu:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1015, '菜单删除', 102, 4, '', '', '', '', 1, 0, 'F', '0', '0', 'system:menu:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1016, '部门查询', 103, 1, '', '', '', '', 1, 0, 'F', '0', '0', 'system:dept:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1017, '部门新增', 103, 2, '', '', '', '', 1, 0, 'F', '0', '0', 'system:dept:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1018, '部门修改', 103, 3, '', '', '', '', 1, 0, 'F', '0', '0', 'system:dept:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1019, '部门删除', 103, 4, '', '', '', '', 1, 0, 'F', '0', '0', 'system:dept:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1020, '岗位查询', 104, 1, '', '', '', '', 1, 0, 'F', '0', '0', 'system:post:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1021, '岗位新增', 104, 2, '', '', '', '', 1, 0, 'F', '0', '0', 'system:post:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1022, '岗位修改', 104, 3, '', '', '', '', 1, 0, 'F', '0', '0', 'system:post:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1023, '岗位删除', 104, 4, '', '', '', '', 1, 0, 'F', '0', '0', 'system:post:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1024, '岗位导出', 104, 5, '', '', '', '', 1, 0, 'F', '0', '0', 'system:post:export', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1025, '字典查询', 105, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:dict:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1026, '字典新增', 105, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:dict:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1027, '字典修改', 105, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:dict:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1028, '字典删除', 105, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:dict:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1029, '字典导出', 105, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:dict:export', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1030, '参数查询', 106, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:config:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1031, '参数新增', 106, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:config:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1032, '参数修改', 106, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:config:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1033, '参数删除', 106, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:config:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1034, '参数导出', 106, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:config:export', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1035, '公告查询', 107, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:notice:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1036, '公告新增', 107, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:notice:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1037, '公告修改', 107, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:notice:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1038, '公告删除', 107, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'system:notice:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1039, '操作查询', 500, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:operlog:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1040, '操作删除', 500, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:operlog:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1041, '日志导出', 500, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:operlog:export', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1042, '登录查询', 501, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:logininfor:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1043, '登录删除', 501, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:logininfor:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1044, '日志导出', 501, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:logininfor:export', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1045, '账户解锁', 501, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:logininfor:unlock', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1046, '在线查询', 109, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:online:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1047, '批量强退', 109, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:online:batchLogout', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1048, '单条强退', 109, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:online:forceLogout', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1049, '任务查询', 110, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:job:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1050, '任务新增', 110, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:job:add', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1051, '任务修改', 110, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:job:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1052, '任务删除', 110, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:job:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1053, '状态修改', 110, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:job:changeStatus', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1054, '任务导出', 110, 6, '#', '', '', '', 1, 0, 'F', '0', '0', 'monitor:job:export', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1055, '生成查询', 116, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'tool:gen:query', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1056, '生成修改', 116, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'tool:gen:edit', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1057, '生成删除', 116, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'tool:gen:remove', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1058, '导入代码', 116, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'tool:gen:import', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1059, '预览代码', 116, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'tool:gen:preview', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (1060, '生成代码', 116, 6, '#', '', '', '', 1, 0, 'F', '0', '0', 'tool:gen:code', '#', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2000, '学习资源管理', 0, 10, 'study', NULL, NULL, '', 1, 0, 'M', '0', '0', '', 'cascader', 'admin', '2025-05-20 16:42:24', 'ry', '2025-05-22 11:12:16', '');
INSERT INTO `sys_menu` VALUES (2001, '图书资源', 2000, 1, 'book', NULL, NULL, '', 1, 0, 'M', '0', '0', NULL, 'documentation', 'admin', '2025-05-21 17:55:31', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2002, '视频资源', 2000, 2, 'video', NULL, NULL, '', 1, 0, 'M', '0', '0', NULL, 'dict', 'admin', '2025-05-21 17:57:42', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2003, '图书分类', 2001, 1, 'book_category', 'cms/book_category/index', NULL, '', 1, 0, 'C', '0', '0', 'cms:book_category:list', '#', 'admin', '2025-05-21 18:19:22', '', NULL, '图书分类菜单');
INSERT INTO `sys_menu` VALUES (2004, '图书分类查询', 2003, 1, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_category:query', '#', 'admin', '2025-05-21 18:19:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2005, '图书分类新增', 2003, 2, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_category:add', '#', 'admin', '2025-05-21 18:19:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2006, '图书分类修改', 2003, 3, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_category:edit', '#', 'admin', '2025-05-21 18:19:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2007, '图书分类删除', 2003, 4, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_category:remove', '#', 'admin', '2025-05-21 18:19:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2008, '图书分类导出', 2003, 5, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_category:export', '#', 'admin', '2025-05-21 18:19:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2021, '图书资源', 2001, 1, 'book_resource', 'cms/book_resource/index', NULL, '', 1, 0, 'C', '0', '0', 'cms:book_resource:list', '#', 'admin', '2025-05-21 18:23:55', '', NULL, '图书资源菜单');
INSERT INTO `sys_menu` VALUES (2022, '图书资源查询', 2021, 1, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_resource:query', '#', 'admin', '2025-05-21 18:23:55', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2023, '图书资源新增', 2021, 2, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_resource:add', '#', 'admin', '2025-05-21 18:23:55', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2024, '图书资源修改', 2021, 3, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_resource:edit', '#', 'admin', '2025-05-21 18:23:55', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2025, '图书资源删除', 2021, 4, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_resource:remove', '#', 'admin', '2025-05-21 18:23:55', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2026, '图书资源导出', 2021, 5, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book_resource:export', '#', 'admin', '2025-05-21 18:23:55', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2027, '视频分类', 2002, 1, 'video_category', 'cms/video_category/index', NULL, '', 1, 0, 'C', '0', '0', 'cms:video_category:list', '#', 'admin', '2025-05-21 18:24:18', '', NULL, '视频分类菜单');
INSERT INTO `sys_menu` VALUES (2028, '视频分类查询', 2027, 1, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_category:query', '#', 'admin', '2025-05-21 18:24:18', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2029, '视频分类新增', 2027, 2, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_category:add', '#', 'admin', '2025-05-21 18:24:18', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2030, '视频分类修改', 2027, 3, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_category:edit', '#', 'admin', '2025-05-21 18:24:18', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2031, '视频分类删除', 2027, 4, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_category:remove', '#', 'admin', '2025-05-21 18:24:18', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2032, '视频分类导出', 2027, 5, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_category:export', '#', 'admin', '2025-05-21 18:24:18', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2033, '视频资源', 2002, 1, 'video_resource', 'cms/video_resource/index', NULL, '', 1, 0, 'C', '0', '0', 'cms:video_resource:list', '#', 'admin', '2025-05-21 18:24:25', '', NULL, '视频资源菜单');
INSERT INTO `sys_menu` VALUES (2034, '视频资源查询', 2033, 1, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_resource:query', '#', 'admin', '2025-05-21 18:24:25', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2035, '视频资源新增', 2033, 2, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_resource:add', '#', 'admin', '2025-05-21 18:24:25', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2036, '视频资源修改', 2033, 3, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_resource:edit', '#', 'admin', '2025-05-21 18:24:25', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2037, '视频资源删除', 2033, 4, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_resource:remove', '#', 'admin', '2025-05-21 18:24:25', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2038, '视频资源导出', 2033, 5, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video_resource:export', '#', 'admin', '2025-05-21 18:24:25', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2039, '图像融合算法', 0, 1, 'fusion', NULL, NULL, '', 1, 0, 'M', '0', '0', NULL, 'list', 'admin', '2025-05-21 18:37:10', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2040, '传统方法', 2039, 1, 'tradition', NULL, NULL, '', 1, 0, 'M', '0', '0', NULL, 'tree', 'admin', '2025-05-21 18:38:38', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2041, '多尺度分解法', 2040, 1, 'multiscale-decomposition-method', NULL, NULL, '', 1, 0, 'M', '0', '0', NULL, 'tree', 'admin', '2025-05-21 18:39:53', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2042, '小波变化法', 2041, 1, 'wavelet-transform-method', 'cms/wavelet-transform-method/index', NULL, '', 1, 0, 'C', '0', '0', '', 'tree', 'admin', '2025-05-21 20:10:14', 'admin', '2025-05-22 19:46:36', '');
INSERT INTO `sys_menu` VALUES (2043, '金字塔法', 2041, 2, 'pyramid-method', 'cms/pyramid-method/index', NULL, '', 1, 0, 'C', '0', '0', '', 'tree', 'admin', '2025-05-21 20:18:42', 'admin', '2025-05-23 03:28:55', '');
INSERT INTO `sys_menu` VALUES (2044, '稀疏表示法', 2040, 2, 'sparse-representation', NULL, NULL, '', 1, 0, 'M', '0', '0', NULL, 'tree', 'admin', '2025-05-21 20:19:57', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2045, '稀疏法', 2044, 1, 'sparse-method', 'cms/sparse-method/index', NULL, '', 1, 0, 'C', '0', '0', '', 'tree', 'admin', '2025-05-21 20:20:45', 'admin', '2025-05-23 03:29:22', '');
INSERT INTO `sys_menu` VALUES (2046, '深度学习法', 2039, 2, 'deep-learning-method', NULL, NULL, '', 1, 0, 'M', '0', '0', NULL, 'tree', 'admin', '2025-05-21 20:21:55', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2047, '卷积神经网络', 2046, 1, 'convolutional-neural-network', 'cms/convolutional-neural-network/index', NULL, '', 1, 0, 'C', '0', '0', '', 'tree', 'admin', '2025-05-21 22:00:36', 'admin', '2025-05-23 03:29:38', '');
INSERT INTO `sys_menu` VALUES (2048, '生成对抗网络', 2046, 2, 'generating-countermeasure-network', 'cms/generating-countermeasure-network/index', NULL, '', 1, 0, 'C', '0', '0', '', 'tree', 'admin', '2025-05-21 22:04:44', 'admin', '2025-05-23 03:29:50', '');
INSERT INTO `sys_menu` VALUES (2049, '自编码器', 2046, 3, 'self-encoder', 'cmd/self-encoder/index', NULL, '', 1, 0, 'C', '0', '0', '', 'tree', 'admin', '2025-05-21 22:05:28', 'admin', '2025-05-23 03:30:01', '');
INSERT INTO `sys_menu` VALUES (2050, '改进方法', 2039, 3, 'improve-method', NULL, NULL, '', 1, 0, 'M', '0', '0', '', 'tree', 'admin', '2025-05-21 22:06:17', 'ry', '2025-05-24 21:27:43', '');
INSERT INTO `sys_menu` VALUES (2051, 'GANResNet', 2050, 1, 'ganresnet', 'cms/ganresnet/index', NULL, '', 1, 0, 'C', '0', '0', '', 'tree', 'admin', '2025-05-21 22:07:33', 'admin', '2025-05-23 03:30:31', '');
INSERT INTO `sys_menu` VALUES (2052, '学习资源', 0, 11, 'study-re', NULL, NULL, '', 1, 0, 'M', '0', '0', '', 'list', 'ry', '2025-05-22 11:13:18', 'admin', '2025-05-24 13:52:16', '');
INSERT INTO `sys_menu` VALUES (2067, '图书', 2052, 1, 'book', 'cms/book/index', NULL, '', 1, 0, 'C', '0', '0', 'cms:book:list', '#', 'admin', '2025-05-22 12:35:22', 'admin', '2025-05-24 13:52:34', '图书菜单');
INSERT INTO `sys_menu` VALUES (2068, '图书查询', 2067, 1, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book:query', '#', 'admin', '2025-05-22 12:35:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2069, '图书新增', 2067, 2, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book:add', '#', 'admin', '2025-05-22 12:35:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2070, '图书修改', 2067, 3, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book:edit', '#', 'admin', '2025-05-22 12:35:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2071, '图书删除', 2067, 4, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book:remove', '#', 'admin', '2025-05-22 12:35:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2072, '图书导出', 2067, 5, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:book:export', '#', 'admin', '2025-05-22 12:35:22', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2073, '视频', 2052, 1, 'video', 'cms/video/index', NULL, '', 1, 0, 'C', '0', '0', 'cms:video:list', '#', 'admin', '2025-05-22 12:35:31', 'admin', '2025-05-24 13:51:03', '视频菜单');
INSERT INTO `sys_menu` VALUES (2074, '视频查询', 2073, 1, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video:query', '#', 'admin', '2025-05-22 12:35:31', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2075, '视频新增', 2073, 2, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video:add', '#', 'admin', '2025-05-22 12:35:31', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2076, '视频修改', 2073, 3, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video:edit', '#', 'admin', '2025-05-22 12:35:31', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2077, '视频删除', 2073, 4, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video:remove', '#', 'admin', '2025-05-22 12:35:31', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2078, '视频导出', 2073, 5, '#', '', NULL, '', 1, 0, 'F', '0', '0', 'cms:video:export', '#', 'admin', '2025-05-22 12:35:31', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2080, '说明书', 0, 21, 'info', 'cmd/info/index', NULL, '', 1, 0, 'C', '0', '0', '', 'tab', 'ry', '2025-05-24 18:53:14', 'admin', '2025-05-25 02:56:36', '');
INSERT INTO `sys_menu` VALUES (2081, '最优寻找', 2039, 10, 'fast-search', NULL, NULL, '', 1, 0, 'M', '0', '0', NULL, 'tree', 'ry', '2025-05-24 21:28:43', '', NULL, '');
INSERT INTO `sys_menu` VALUES (2082, '最优寻找方法', 2081, 1, 'fast-search-method', 'cms/fast-search-method/index', '', '', 1, 0, 'C', '0', '0', '', 'tree', 'admin', '2025-05-24 21:30:40', 'admin', '2025-05-24 22:43:19', '');
INSERT INTO `sys_menu` VALUES (2083, '图书详情', 2052, 2, 'book-detail', 'cms/book/detail', 'id', 'BookDetail', 1, 0, 'C', '1', '0', '', 'tab', 'admin', '2025-05-25 10:38:55', 'admin', '2025-05-25 16:55:00', '');
INSERT INTO `sys_menu` VALUES (2084, '视频详情', 2052, 2, 'video-detail', 'cms/video/detail', 'id', 'VideoDetail', 1, 0, 'C', '1', '0', NULL, 'tab', 'admin', '2025-05-25 10:41:22', '', NULL, '');

-- ----------------------------
-- Table structure for sys_notice
-- ----------------------------
DROP TABLE IF EXISTS `sys_notice`;
CREATE TABLE `sys_notice`  (
  `notice_id` int NOT NULL AUTO_INCREMENT COMMENT '公告ID',
  `notice_title` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '公告标题',
  `notice_type` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '公告类型（1通知 2公告）',
  `notice_content` longblob NULL COMMENT '公告内容',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '公告状态（0正常 1关闭）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`notice_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 10 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '通知公告表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_notice
-- ----------------------------
INSERT INTO `sys_notice` VALUES (1, '温馨提醒：2018-07-01 若依新版本发布啦', '2', 0xE696B0E78988E69CACE58685E5AEB9, '0', 'admin', '2025-05-17 15:01:03', '', NULL, '管理员');
INSERT INTO `sys_notice` VALUES (2, '维护通知：2018-07-01 若依系统凌晨维护', '1', 0xE7BBB4E68AA4E58685E5AEB9, '0', 'admin', '2025-05-17 15:01:03', '', NULL, '管理员');

-- ----------------------------
-- Table structure for sys_oper_log
-- ----------------------------
DROP TABLE IF EXISTS `sys_oper_log`;
CREATE TABLE `sys_oper_log`  (
  `oper_id` bigint NOT NULL AUTO_INCREMENT COMMENT '日志主键',
  `title` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '模块标题',
  `business_type` int NULL DEFAULT 0 COMMENT '业务类型（0其它 1新增 2修改 3删除）',
  `method` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '方法名称',
  `request_method` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '请求方式',
  `operator_type` int NULL DEFAULT 0 COMMENT '操作类别（0其它 1后台用户 2手机端用户）',
  `oper_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '操作人员',
  `dept_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '部门名称',
  `oper_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '请求URL',
  `oper_ip` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '主机地址',
  `oper_location` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '操作地点',
  `oper_param` varchar(2000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '请求参数',
  `json_result` varchar(2000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '返回参数',
  `status` int NULL DEFAULT 0 COMMENT '操作状态（0正常 1异常）',
  `error_msg` varchar(2000) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '错误消息',
  `oper_time` datetime NULL DEFAULT NULL COMMENT '操作时间',
  `cost_time` bigint NULL DEFAULT 0 COMMENT '消耗时间',
  PRIMARY KEY (`oper_id`) USING BTREE,
  INDEX `idx_sys_oper_log_bt`(`business_type` ASC) USING BTREE,
  INDEX `idx_sys_oper_log_s`(`status` ASC) USING BTREE,
  INDEX `idx_sys_oper_log_ot`(`oper_time` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 212 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '操作日志记录' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_oper_log
-- ----------------------------
INSERT INTO `sys_oper_log` VALUES (100, '个人信息', 2, 'com.ruoyi.web.controller.system.SysProfileController.updateProfile()', 'PUT', 1, 'ry', '测试部门', '/system/user/profile', '127.0.0.1', '内网IP', '{\"admin\":false,\"email\":\"ry@qq.com\",\"nickName\":\"若依\",\"params\":{},\"phonenumber\":\"15666666666\",\"sex\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-20 14:51:15', 17);
INSERT INTO `sys_oper_log` VALUES (101, '字典类型', 9, 'com.ruoyi.web.controller.system.SysDictTypeController.refreshCache()', 'DELETE', 1, 'ry', '测试部门', '/system/dict/type/refreshCache', '127.0.0.1', '内网IP', '', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-20 14:51:31', 35);
INSERT INTO `sys_oper_log` VALUES (102, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"cascader\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"学习资源\",\"menuType\":\"M\",\"orderNum\":10,\"params\":{},\"parentId\":0,\"path\":\"manage\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-20 16:42:24', 30);
INSERT INTO `sys_oper_log` VALUES (103, '代码生成', 6, 'com.ruoyi.generator.controller.GenController.importTableSave()', 'POST', 1, 'admin', '研发部门', '/tool/gen/importTable', '127.0.0.1', '内网IP', '{\"tables\":\"book_resource,book_category,video_category,video_resource\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-20 16:43:14', 159);
INSERT INTO `sys_oper_log` VALUES (104, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"resource\",\"className\":\"BookResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnComment\":\"书编号\",\"columnId\":3,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"书籍名称\",\"columnId\":4,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":5,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"usableColumn\":false},{\"capJavaField\":\"FileType\",\"columnComment\":\"文件类型，如 PDF、EPUB\",\"columnId\":6,\"columnName\":\"file_type\",\"columnType\":\"varchar(50)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"select\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\"', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 17:54:23', 56);
INSERT INTO `sys_oper_log` VALUES (105, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"documentation\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"图书资源\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2000,\"path\":\"book\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 17:55:31', 16);
INSERT INTO `sys_oper_log` VALUES (106, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-20 16:42:24\",\"icon\":\"cascader\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2000,\"menuName\":\"学习资源\",\"menuType\":\"M\",\"orderNum\":10,\"params\":{},\"parentId\":0,\"path\":\"study\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 17:56:12', 11);
INSERT INTO `sys_oper_log` VALUES (107, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"dict\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"视频资源\",\"menuType\":\"M\",\"orderNum\":2,\"params\":{},\"parentId\":2000,\"path\":\"video\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 17:57:43', 11);
INSERT INTO `sys_oper_log` VALUES (108, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"resource\",\"className\":\"BookResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnComment\":\"书编号\",\"columnId\":3,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 17:54:23\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"书籍名称\",\"columnId\":4,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 17:54:23\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":5,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 17:54:23\",\"usableColumn\":false},{\"capJavaField\":\"FileType\",\"columnComment\":\"文件类型，如 PDF、EPUB\",\"columnId\":6,\"columnName\":\"file_type\",\"columnType\":\"varchar(50)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"select\",\"increment\":false,\"inse', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 17:58:21', 30);
INSERT INTO `sys_oper_log` VALUES (109, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"category\",\"className\":\"BookCategory\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":1,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":1,\"updateBy\":\"\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"分类名称，如基础理论、算法实践\",\"columnId\":2,\"columnName\":\"name\",\"columnType\":\"varchar(100)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":1,\"updateBy\":\"\",\"usableColumn\":false}],\"crud\":true,\"functionAuthor\":\"lhl\",\"functionName\":\"book_category\",\"genPath\":\"/\",\"genType\":\"0\",\"moduleName\":\"cms\",\"options\":\"{\\\"parentMenuId\\\":2001}\",\"packageName\":\"com.ruoyi.cms\",\"params\":{\"parentMenuId\":2001},\"parentMenuId\":2001,\"sub\":false,\"tableComment\":\"图书分类\",\"tableId\":1,\"tableName\":\"book_category\",\"tplCategory\":\"crud\",\"tplWebType\":\"element-plus\",\"tree\":false}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 17:59:24', 13);
INSERT INTO `sys_oper_log` VALUES (110, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"category\",\"className\":\"VideoCategory\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":11,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":3,\"updateBy\":\"\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"分类名称，如多尺度分解、稀疏表示、深度学习\",\"columnId\":12,\"columnName\":\"name\",\"columnType\":\"varchar(100)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":3,\"updateBy\":\"\",\"usableColumn\":false}],\"crud\":true,\"functionAuthor\":\"lhl\",\"functionName\":\"video_category\",\"genPath\":\"/\",\"genType\":\"0\",\"moduleName\":\"cms\",\"options\":\"{\\\"parentMenuId\\\":2002}\",\"packageName\":\"com.ruoyi.cms\",\"params\":{\"parentMenuId\":2002},\"parentMenuId\":2002,\"sub\":false,\"tableComment\":\"视频分类\",\"tableId\":3,\"tableName\":\"video_category\",\"tplCategory\":\"crud\",\"tplWebType\":\"element-plus\",\"tree\":false}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:00:07', 14);
INSERT INTO `sys_oper_log` VALUES (111, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"resource\",\"className\":\"VideoResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":13,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"视频标题\",\"columnId\":14,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":15,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"usableColumn\":false},{\"capJavaField\":\"Duration\",\"columnComment\":\"视频时长\",\"columnId\":16,\"columnName\":\"duration\",\"columnType\":\"time\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"datetime\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"duration\",\"javaT', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:01:03', 28);
INSERT INTO `sys_oper_log` VALUES (112, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"book_resource\",\"className\":\"BookResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnComment\":\"书编号\",\"columnId\":3,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 17:58:21\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"书籍名称\",\"columnId\":4,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 17:58:21\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":5,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 17:58:21\",\"usableColumn\":false},{\"capJavaField\":\"FileType\",\"columnComment\":\"文件类型，如 PDF、EPUB\",\"columnId\":6,\"columnName\":\"file_type\",\"columnType\":\"varchar(50)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"select\",\"increment\":false,', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:01:33', 25);
INSERT INTO `sys_oper_log` VALUES (113, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"book_category\",\"className\":\"BookCategory\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":1,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":1,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 17:59:24\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"分类名称，如基础理论、算法实践\",\"columnId\":2,\"columnName\":\"name\",\"columnType\":\"varchar(100)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":1,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 17:59:24\",\"usableColumn\":false}],\"crud\":true,\"functionAuthor\":\"lhl\",\"functionName\":\"图书分类\",\"genPath\":\"/\",\"genType\":\"0\",\"moduleName\":\"cms\",\"options\":\"{\\\"parentMenuId\\\":2001}\",\"packageName\":\"com.ruoyi.cms\",\"params\":{\"parentMenuId\":2001},\"parentMenuId\":2001,\"sub\":false,\"tableComment\":\"图书分类\",\"tableId\":1,\"tableName\":\"book_category\",\"tplCategory\":\"crud\",\"tplWebType\":\"element-plus\",\"tree\":false}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:01:47', 12);
INSERT INTO `sys_oper_log` VALUES (114, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"video_category\",\"className\":\"VideoCategory\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":11,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":3,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:00:07\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"分类名称，如多尺度分解、稀疏表示、深度学习\",\"columnId\":12,\"columnName\":\"name\",\"columnType\":\"varchar(100)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":3,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:00:07\",\"usableColumn\":false}],\"crud\":true,\"functionAuthor\":\"lhl\",\"functionName\":\"视频分类\",\"genPath\":\"/\",\"genType\":\"0\",\"moduleName\":\"cms\",\"options\":\"{\\\"parentMenuId\\\":2002}\",\"packageName\":\"com.ruoyi.cms\",\"params\":{\"parentMenuId\":2002},\"parentMenuId\":2002,\"sub\":false,\"tableComment\":\"视频分类\",\"tableId\":3,\"tableName\":\"video_category\",\"tplCategory\":\"crud\",\"tplWebType\":\"element-plus\",\"tree\":false}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:01:58', 11);
INSERT INTO `sys_oper_log` VALUES (115, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"video_resource\",\"className\":\"VideoResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":13,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:01:03\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"视频标题\",\"columnId\":14,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:01:03\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":15,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:01:03\",\"usableColumn\":false},{\"capJavaField\":\"Duration\",\"columnComment\":\"视频时长\",\"columnId\":16,\"columnName\":\"duration\",\"columnType\":\"time\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"datetime\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isInc', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:02:25', 28);
INSERT INTO `sys_oper_log` VALUES (116, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"video_resource\",\"className\":\"VideoResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":13,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":false,\"isIncrement\":\"1\",\"isInsert\":\"0\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:02:25\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"视频标题\",\"columnId\":14,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:02:25\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":15,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:02:25\",\"usableColumn\":false},{\"capJavaField\":\"Duration\",\"columnComment\":\"视频时长\",\"columnId\":16,\"columnName\":\"duration\",\"columnType\":\"time\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"datetime\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIn', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:03:13', 27);
INSERT INTO `sys_oper_log` VALUES (117, '代码生成', 8, 'com.ruoyi.generator.controller.GenController.batchGenCode()', 'GET', 1, 'admin', '研发部门', '/tool/gen/batchGenCode', '127.0.0.1', '内网IP', '{\"tables\":\"book_category,book_resource,video_category,video_resource\"}', NULL, 0, NULL, '2025-05-21 18:17:18', 1011);
INSERT INTO `sys_oper_log` VALUES (118, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"list\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"图像融合算法\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":0,\"path\":\"fusion\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:37:10', 109);
INSERT INTO `sys_oper_log` VALUES (119, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"传统方法\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2039,\"path\":\"tradition\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:38:38', 13);
INSERT INTO `sys_oper_log` VALUES (120, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"多尺度分解法\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2040,\"path\":\"multiscale-decomposition-method\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 18:39:53', 14);
INSERT INTO `sys_oper_log` VALUES (121, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"小波变化法\",\"menuType\":\"M\",\"orderNum\":2,\"params\":{},\"parentId\":2041,\"path\":\"wavelet-transform-method\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 20:10:14', 10);
INSERT INTO `sys_oper_log` VALUES (122, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-21 20:10:14\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2042,\"menuName\":\"小波变化法\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2041,\"path\":\"wavelet-transform-method\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 20:10:24', 10);
INSERT INTO `sys_oper_log` VALUES (123, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"金字塔法\",\"menuType\":\"M\",\"orderNum\":2,\"params\":{},\"parentId\":2041,\"path\":\"jinzita-method\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 20:18:42', 13);
INSERT INTO `sys_oper_log` VALUES (124, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"稀疏表示法\",\"menuType\":\"M\",\"orderNum\":2,\"params\":{},\"parentId\":2040,\"path\":\"sparse-representation\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 20:19:57', 10);
INSERT INTO `sys_oper_log` VALUES (125, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"稀疏法\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2044,\"path\":\"sparse-method\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 20:20:45', 11);
INSERT INTO `sys_oper_log` VALUES (126, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"深度学习法\",\"menuType\":\"M\",\"orderNum\":2,\"params\":{},\"parentId\":2039,\"path\":\"deep-learning-method\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 20:21:55', 12);
INSERT INTO `sys_oper_log` VALUES (127, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"卷积神经网络\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2046,\"path\":\"convolutional-neural-network\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 22:00:36', 31);
INSERT INTO `sys_oper_log` VALUES (128, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"生成对抗网络\",\"menuType\":\"M\",\"orderNum\":2,\"params\":{},\"parentId\":2046,\"path\":\"generating-countermeasure-network\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 22:04:44', 10);
INSERT INTO `sys_oper_log` VALUES (129, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"自编码器\",\"menuType\":\"M\",\"orderNum\":3,\"params\":{},\"parentId\":2046,\"path\":\"self-encoder\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 22:05:28', 9);
INSERT INTO `sys_oper_log` VALUES (130, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"改进方法\",\"menuType\":\"M\",\"orderNum\":3,\"params\":{},\"parentId\":2039,\"path\":\"improve-method\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 22:06:17', 10);
INSERT INTO `sys_oper_log` VALUES (131, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"GANResNet\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2050,\"path\":\"ganresnet\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-21 22:07:33', 8);
INSERT INTO `sys_oper_log` VALUES (132, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'ry', '测试部门', '/system/menu/2051', '127.0.0.1', '内网IP', '2051', '{\"msg\":\"菜单已分配,不允许删除\",\"code\":601}', 0, NULL, '2025-05-22 11:09:58', 9);
INSERT INTO `sys_oper_log` VALUES (133, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-20 16:42:24\",\"icon\":\"cascader\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2000,\"menuName\":\"学习资源管理\",\"menuType\":\"M\",\"orderNum\":10,\"params\":{},\"parentId\":0,\"path\":\"study\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"ry\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 11:12:16', 16);
INSERT INTO `sys_oper_log` VALUES (134, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"ry\",\"icon\":\"list\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"学习资源\",\"menuType\":\"M\",\"orderNum\":11,\"params\":{},\"parentId\":0,\"path\":\"study-re\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 11:13:18', 9);
INSERT INTO `sys_oper_log` VALUES (135, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"icon\":\"list\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"学习资源\",\"menuType\":\"M\",\"orderNum\":11,\"params\":{},\"parentId\":0,\"path\":\"study-resource\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"新增菜单\'学习资源\'失败，菜单名称已存在\",\"code\":500}', 0, NULL, '2025-05-22 11:14:12', 5);
INSERT INTO `sys_oper_log` VALUES (136, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"图书\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2052,\"path\":\"book\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 11:35:14', 11);
INSERT INTO `sys_oper_log` VALUES (137, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"视频\",\"menuType\":\"M\",\"orderNum\":2,\"params\":{},\"parentId\":2052,\"path\":\"video\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 11:36:30', 11);
INSERT INTO `sys_oper_log` VALUES (138, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-22 11:35:14\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2053,\"menuName\":\"图书\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2052,\"path\":\"book\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 11:36:35', 12);
INSERT INTO `sys_oper_log` VALUES (139, '代码生成', 8, 'com.ruoyi.generator.controller.GenController.batchGenCode()', 'GET', 1, 'admin', '研发部门', '/tool/gen/batchGenCode', '127.0.0.1', '内网IP', '{\"tables\":\"book_resource\"}', NULL, 0, NULL, '2025-05-22 12:24:40', 746);
INSERT INTO `sys_oper_log` VALUES (140, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"book\",\"className\":\"BookResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnComment\":\"书编号\",\"columnId\":3,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:01:33\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"书籍名称\",\"columnId\":4,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:01:33\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":5,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:01:33\",\"usableColumn\":false},{\"capJavaField\":\"FileType\",\"columnComment\":\"文件类型，如 PDF、EPUB\",\"columnId\":6,\"columnName\":\"file_type\",\"columnType\":\"varchar(50)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"select\",\"increment\":false,\"insert\":', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:28:20', 66);
INSERT INTO `sys_oper_log` VALUES (141, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"video\",\"className\":\"VideoResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":13,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":false,\"isIncrement\":\"1\",\"isInsert\":\"0\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:03:13\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"视频标题\",\"columnId\":14,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:03:13\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":15,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-21 18:03:13\",\"usableColumn\":false},{\"capJavaField\":\"Duration\",\"columnComment\":\"视频时长\",\"columnId\":16,\"columnName\":\"duration\",\"columnType\":\"time\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"datetime\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:28:43', 28);
INSERT INTO `sys_oper_log` VALUES (142, '代码生成', 8, 'com.ruoyi.generator.controller.GenController.batchGenCode()', 'GET', 1, 'admin', '研发部门', '/tool/gen/batchGenCode', '127.0.0.1', '内网IP', '{\"tables\":\"video_resource,book_resource\"}', NULL, 0, NULL, '2025-05-22 12:28:48', 74);
INSERT INTO `sys_oper_log` VALUES (143, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"book\",\"className\":\"BookResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnComment\":\"书编号\",\"columnId\":3,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:28:20\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"书籍名称\",\"columnId\":4,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:28:20\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":5,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:28:20\",\"usableColumn\":false},{\"capJavaField\":\"FileType\",\"columnComment\":\"文件类型，如 PDF、EPUB\",\"columnId\":6,\"columnName\":\"file_type\",\"columnType\":\"varchar(50)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"select\",\"increment\":false,\"insert\":', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:32:18', 24);
INSERT INTO `sys_oper_log` VALUES (144, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"video\",\"className\":\"VideoResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":13,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":false,\"isIncrement\":\"1\",\"isInsert\":\"0\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:28:43\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"视频标题\",\"columnId\":14,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:28:43\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":15,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:28:43\",\"usableColumn\":false},{\"capJavaField\":\"Duration\",\"columnComment\":\"视频时长\",\"columnId\":16,\"columnName\":\"duration\",\"columnType\":\"time\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"datetime\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:32:26', 23);
INSERT INTO `sys_oper_log` VALUES (145, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"video\",\"className\":\"VideoResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnId\":13,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":false,\"isIncrement\":\"1\",\"isInsert\":\"0\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:32:26\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"视频标题\",\"columnId\":14,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:32:26\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":15,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":4,\"updateBy\":\"\",\"updateTime\":\"2025-05-22 12:32:26\",\"usableColumn\":false},{\"capJavaField\":\"Duration\",\"columnComment\":\"视频时长\",\"columnId\":16,\"columnName\":\"duration\",\"columnType\":\"time\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"datetime\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:32:43', 26);
INSERT INTO `sys_oper_log` VALUES (146, '代码生成', 8, 'com.ruoyi.generator.controller.GenController.batchGenCode()', 'GET', 1, 'admin', '研发部门', '/tool/gen/batchGenCode', '127.0.0.1', '内网IP', '{\"tables\":\"video_resource,book_resource\"}', NULL, 0, NULL, '2025-05-22 12:32:49', 88);
INSERT INTO `sys_oper_log` VALUES (147, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2053', '127.0.0.1', '内网IP', '2053', '{\"msg\":\"存在子菜单,不允许删除\",\"code\":601}', 0, NULL, '2025-05-22 12:33:05', 2);
INSERT INTO `sys_oper_log` VALUES (148, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2055', '127.0.0.1', '内网IP', '2055', '{\"msg\":\"存在子菜单,不允许删除\",\"code\":601}', 0, NULL, '2025-05-22 12:33:12', 2);
INSERT INTO `sys_oper_log` VALUES (149, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2056', '127.0.0.1', '内网IP', '2056', '{\"msg\":\"菜单已分配,不允许删除\",\"code\":601}', 0, NULL, '2025-05-22 12:33:16', 3);
INSERT INTO `sys_oper_log` VALUES (150, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2056', '127.0.0.1', '内网IP', '2056', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:22', 14);
INSERT INTO `sys_oper_log` VALUES (151, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2055', '127.0.0.1', '内网IP', '2055', '{\"msg\":\"存在子菜单,不允许删除\",\"code\":601}', 0, NULL, '2025-05-22 12:34:25', 3);
INSERT INTO `sys_oper_log` VALUES (152, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2057', '127.0.0.1', '内网IP', '2057', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:29', 7);
INSERT INTO `sys_oper_log` VALUES (153, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2058', '127.0.0.1', '内网IP', '2058', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:30', 9);
INSERT INTO `sys_oper_log` VALUES (154, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2059', '127.0.0.1', '内网IP', '2059', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:31', 8);
INSERT INTO `sys_oper_log` VALUES (155, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2060', '127.0.0.1', '内网IP', '2060', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:32', 10);
INSERT INTO `sys_oper_log` VALUES (156, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2055', '127.0.0.1', '内网IP', '2055', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:33', 13);
INSERT INTO `sys_oper_log` VALUES (157, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2053', '127.0.0.1', '内网IP', '2053', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:34', 10);
INSERT INTO `sys_oper_log` VALUES (158, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2062', '127.0.0.1', '内网IP', '2062', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:37', 8);
INSERT INTO `sys_oper_log` VALUES (159, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2063', '127.0.0.1', '内网IP', '2063', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:38', 11);
INSERT INTO `sys_oper_log` VALUES (160, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2064', '127.0.0.1', '内网IP', '2064', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:38', 8);
INSERT INTO `sys_oper_log` VALUES (161, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2065', '127.0.0.1', '内网IP', '2065', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:40', 7);
INSERT INTO `sys_oper_log` VALUES (162, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2066', '127.0.0.1', '内网IP', '2066', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:40', 8);
INSERT INTO `sys_oper_log` VALUES (163, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2061', '127.0.0.1', '内网IP', '2061', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:41', 9);
INSERT INTO `sys_oper_log` VALUES (164, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2054', '127.0.0.1', '内网IP', '2054', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 12:34:43', 11);
INSERT INTO `sys_oper_log` VALUES (165, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/wavelet-transform-method/index\",\"createTime\":\"2025-05-21 20:10:14\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2042,\"menuName\":\"小波变化法\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2041,\"path\":\"wavelet-transform-method\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-22 19:46:36', 21);
INSERT INTO `sys_oper_log` VALUES (166, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/jinzita-method/index\",\"createTime\":\"2025-05-21 20:18:42\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2043,\"menuName\":\"金字塔法\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2041,\"path\":\"jinzita-method\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:27:43', 27);
INSERT INTO `sys_oper_log` VALUES (167, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/jinzita-method/index\",\"createTime\":\"2025-05-21 20:18:42\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2043,\"menuName\":\"金字塔法\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2041,\"path\":\"jinzita-method\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:28:01', 13);
INSERT INTO `sys_oper_log` VALUES (168, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/pyramid-method/index\",\"createTime\":\"2025-05-21 20:18:42\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2043,\"menuName\":\"金字塔法\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2041,\"path\":\"pyramid-method\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:28:55', 14);
INSERT INTO `sys_oper_log` VALUES (169, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/sparse-method/index\",\"createTime\":\"2025-05-21 20:20:45\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2045,\"menuName\":\"稀疏法\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2044,\"path\":\"sparse-method\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:29:22', 11);
INSERT INTO `sys_oper_log` VALUES (170, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/convolutional-neural-network/index\",\"createTime\":\"2025-05-21 22:00:36\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2047,\"menuName\":\"卷积神经网络\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2046,\"path\":\"convolutional-neural-network\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:29:38', 11);
INSERT INTO `sys_oper_log` VALUES (171, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/generating-countermeasure-network/index\",\"createTime\":\"2025-05-21 22:04:44\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2048,\"menuName\":\"生成对抗网络\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2046,\"path\":\"generating-countermeasure-network\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:29:50', 13);
INSERT INTO `sys_oper_log` VALUES (172, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cmd/self-encoder/index\",\"createTime\":\"2025-05-21 22:05:28\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2049,\"menuName\":\"自编码器\",\"menuType\":\"C\",\"orderNum\":3,\"params\":{},\"parentId\":2046,\"path\":\"self-encoder\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:30:01', 14);
INSERT INTO `sys_oper_log` VALUES (173, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-21 22:06:17\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2050,\"menuName\":\"最优方法\",\"menuType\":\"M\",\"orderNum\":3,\"params\":{},\"parentId\":2039,\"path\":\"improve-method\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:30:15', 19);
INSERT INTO `sys_oper_log` VALUES (174, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/ganresnet/index\",\"createTime\":\"2025-05-21 22:07:33\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2051,\"menuName\":\"GANResNet\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2050,\"path\":\"ganresnet\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:30:31', 13);
INSERT INTO `sys_oper_log` VALUES (175, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cmd/optimal/index\",\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"最优算法\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2050,\"path\":\"optimal\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-23 03:31:40', 12);
INSERT INTO `sys_oper_log` VALUES (176, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/book/index\",\"createTime\":\"2025-05-22 12:35:22\",\"icon\":\"#\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2067,\"menuName\":\"图书\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2052,\"path\":\"book\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 13:49:49', 36);
INSERT INTO `sys_oper_log` VALUES (177, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/video/index\",\"createTime\":\"2025-05-22 12:35:31\",\"icon\":\"#\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2073,\"menuName\":\"视频\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2052,\"path\":\"video\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 13:49:56', 11);
INSERT INTO `sys_oper_log` VALUES (178, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/book/index\",\"createTime\":\"2025-05-22 12:35:22\",\"icon\":\"#\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2067,\"menuName\":\"图书\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2052,\"path\":\"book\",\"perms\":\"cms:video:list\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 13:50:52', 13);
INSERT INTO `sys_oper_log` VALUES (179, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/video/index\",\"createTime\":\"2025-05-22 12:35:31\",\"icon\":\"#\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2073,\"menuName\":\"视频\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2052,\"path\":\"video\",\"perms\":\"cms:video:list\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 13:51:03', 21);
INSERT INTO `sys_oper_log` VALUES (180, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-22 11:13:18\",\"icon\":\"list\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2052,\"menuName\":\"学习资源\",\"menuType\":\"M\",\"orderNum\":11,\"params\":{},\"parentId\":0,\"path\":\"study-re\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 13:52:16', 11);
INSERT INTO `sys_oper_log` VALUES (181, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/book/index\",\"createTime\":\"2025-05-22 12:35:22\",\"icon\":\"#\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2067,\"menuName\":\"图书\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2052,\"path\":\"book\",\"perms\":\"cms:book:list\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 13:52:34', 14);
INSERT INTO `sys_oper_log` VALUES (182, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"ry\",\"icon\":\"tab\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"说明书\",\"menuType\":\"M\",\"orderNum\":21,\"params\":{},\"parentId\":0,\"path\":\"info\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 18:53:14', 42);
INSERT INTO `sys_oper_log` VALUES (183, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'ry', '测试部门', '/system/menu/4', '127.0.0.1', '内网IP', '4', '{\"msg\":\"菜单已分配,不允许删除\",\"code\":601}', 0, NULL, '2025-05-24 18:53:26', 11);
INSERT INTO `sys_oper_log` VALUES (184, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/4', '127.0.0.1', '内网IP', '4', '{\"msg\":\"菜单已分配,不允许删除\",\"code\":601}', 0, NULL, '2025-05-24 18:53:51', 9);
INSERT INTO `sys_oper_log` VALUES (185, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-17 15:01:03\",\"icon\":\"guide\",\"isCache\":\"0\",\"isFrame\":\"0\",\"menuId\":4,\"menuName\":\"若依官网\",\"menuType\":\"M\",\"orderNum\":4,\"params\":{},\"parentId\":0,\"path\":\"http://ruoyi.vip\",\"perms\":\"\",\"query\":\"\",\"routeName\":\"\",\"status\":\"1\",\"updateBy\":\"ry\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 18:54:38', 15);
INSERT INTO `sys_oper_log` VALUES (186, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-17 15:01:03\",\"icon\":\"guide\",\"isCache\":\"0\",\"isFrame\":\"0\",\"menuId\":4,\"menuName\":\"若依官网\",\"menuType\":\"M\",\"orderNum\":4,\"params\":{},\"parentId\":0,\"path\":\"http://ruoyi.vip\",\"perms\":\"\",\"query\":\"\",\"routeName\":\"\",\"status\":\"1\",\"updateBy\":\"ry\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 18:54:44', 15);
INSERT INTO `sys_oper_log` VALUES (187, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"system/dept/index\",\"createTime\":\"2025-05-17 15:01:03\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":103,\"menuName\":\"部门管理\",\"menuType\":\"C\",\"orderNum\":4,\"params\":{},\"parentId\":1,\"path\":\"dept\",\"perms\":\"system:dept:list\",\"query\":\"\",\"routeName\":\"\",\"status\":\"1\",\"updateBy\":\"ry\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 18:56:05', 15);
INSERT INTO `sys_oper_log` VALUES (188, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"system/post/index\",\"createTime\":\"2025-05-17 15:01:03\",\"icon\":\"post\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":104,\"menuName\":\"岗位管理\",\"menuType\":\"C\",\"orderNum\":5,\"params\":{},\"parentId\":1,\"path\":\"post\",\"perms\":\"system:post:list\",\"query\":\"\",\"routeName\":\"\",\"status\":\"1\",\"updateBy\":\"ry\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 18:56:10', 13);
INSERT INTO `sys_oper_log` VALUES (189, '个人信息', 2, 'com.ruoyi.web.controller.system.SysProfileController.updateProfile()', 'PUT', 1, 'ry', '测试部门', '/system/user/profile', '127.0.0.1', '内网IP', '{\"admin\":false,\"email\":\"ry@qq.com\",\"nickName\":\"admin\",\"params\":{},\"phonenumber\":\"15666666666\",\"sex\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 19:06:50', 29);
INSERT INTO `sys_oper_log` VALUES (190, '个人信息', 2, 'com.ruoyi.web.controller.system.SysProfileController.updateProfile()', 'PUT', 1, 'ry', '测试部门', '/system/user/profile', '127.0.0.1', '内网IP', '{\"admin\":false,\"email\":\"ry@qq.com\",\"nickName\":\"笑天菜\",\"params\":{},\"phonenumber\":\"15666666666\",\"sex\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 19:10:34', 19);
INSERT INTO `sys_oper_log` VALUES (191, '用户头像', 2, 'com.ruoyi.web.controller.system.SysProfileController.avatar()', 'POST', 1, 'ry', '测试部门', '/system/user/profile/avatar', '127.0.0.1', '内网IP', '', '{\"msg\":\"操作成功\",\"imgUrl\":\"/profile/avatar/2025/05/24/DC_2012-08-06_28209_20250524191053A016.jpg\",\"code\":200}', 0, NULL, '2025-05-24 19:10:53', 34);
INSERT INTO `sys_oper_log` VALUES (192, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createTime\":\"2025-05-21 22:06:17\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2050,\"menuName\":\"改进方法\",\"menuType\":\"M\",\"orderNum\":3,\"params\":{},\"parentId\":2039,\"path\":\"improve-method\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"ry\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 21:27:43', 27);
INSERT INTO `sys_oper_log` VALUES (193, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'ry', '测试部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"ry\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"最优寻找\",\"menuType\":\"M\",\"orderNum\":10,\"params\":{},\"parentId\":2039,\"path\":\"fast-search\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 21:28:43', 19);
INSERT INTO `sys_oper_log` VALUES (194, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"最优寻找方法\",\"menuType\":\"M\",\"orderNum\":1,\"params\":{},\"parentId\":2081,\"path\":\"fast-search-method\",\"status\":\"0\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 21:30:40', 15);
INSERT INTO `sys_oper_log` VALUES (195, '菜单管理', 3, 'com.ruoyi.web.controller.system.SysMenuController.remove()', 'DELETE', 1, 'admin', '研发部门', '/system/menu/2079', '127.0.0.1', '内网IP', '2079', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 21:31:12', 18);
INSERT INTO `sys_oper_log` VALUES (196, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"fast-search-method\",\"createTime\":\"2025-05-24 21:30:40\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2082,\"menuName\":\"最优寻找方法\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2081,\"path\":\"fast-search-method\",\"perms\":\"\",\"query\":\"fast-search-method\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 21:31:25', 13);
INSERT INTO `sys_oper_log` VALUES (197, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/fast-search-method\",\"createTime\":\"2025-05-24 21:30:40\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2082,\"menuName\":\"最优寻找方法\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2081,\"path\":\"fast-search-method\",\"perms\":\"\",\"query\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 21:31:43', 14);
INSERT INTO `sys_oper_log` VALUES (198, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/fast-search-method/index\",\"createTime\":\"2025-05-24 21:30:40\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2082,\"menuName\":\"最优寻找方法\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2081,\"path\":\"fast-search-method\",\"perms\":\"\",\"query\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 21:32:16', 12);
INSERT INTO `sys_oper_log` VALUES (199, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/fast-search-method/index\",\"createTime\":\"2025-05-24 21:30:40\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2082,\"menuName\":\"最优寻找方法\",\"menuType\":\"C\",\"orderNum\":1,\"params\":{},\"parentId\":2081,\"path\":\"fast-search-method\",\"perms\":\"\",\"query\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-24 22:43:19', 14);
INSERT INTO `sys_oper_log` VALUES (200, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.synchDb()', 'GET', 1, 'admin', '研发部门', '/tool/gen/synchDb/book_resource', '127.0.0.1', '内网IP', '{}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 02:52:26', 115);
INSERT INTO `sys_oper_log` VALUES (201, '代码生成', 2, 'com.ruoyi.generator.controller.GenController.editSave()', 'PUT', 1, 'admin', '研发部门', '/tool/gen', '127.0.0.1', '内网IP', '{\"businessName\":\"book\",\"className\":\"BookResource\",\"columns\":[{\"capJavaField\":\"Id\",\"columnComment\":\"\",\"columnId\":3,\"columnName\":\"id\",\"columnType\":\"int\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":false,\"htmlType\":\"input\",\"increment\":true,\"insert\":true,\"isIncrement\":\"1\",\"isInsert\":\"1\",\"isPk\":\"1\",\"isRequired\":\"0\",\"javaField\":\"id\",\"javaType\":\"Long\",\"list\":false,\"params\":{},\"pk\":true,\"query\":false,\"queryType\":\"EQ\",\"required\":false,\"sort\":1,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-25 02:52:26\",\"usableColumn\":false},{\"capJavaField\":\"Name\",\"columnComment\":\"书籍名称\",\"columnId\":4,\"columnName\":\"name\",\"columnType\":\"varchar(200)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"input\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"name\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"LIKE\",\"required\":true,\"sort\":2,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-25 02:52:26\",\"usableColumn\":false},{\"capJavaField\":\"StoragePath\",\"columnComment\":\"文件存储路径\",\"columnId\":5,\"columnName\":\"storage_path\",\"columnType\":\"varchar(500)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"textarea\",\"increment\":false,\"insert\":true,\"isEdit\":\"1\",\"isIncrement\":\"0\",\"isInsert\":\"1\",\"isList\":\"1\",\"isPk\":\"0\",\"isQuery\":\"1\",\"isRequired\":\"1\",\"javaField\":\"storagePath\",\"javaType\":\"String\",\"list\":true,\"params\":{},\"pk\":false,\"query\":true,\"queryType\":\"EQ\",\"required\":true,\"sort\":3,\"superColumn\":false,\"tableId\":2,\"updateBy\":\"\",\"updateTime\":\"2025-05-25 02:52:26\",\"usableColumn\":false},{\"capJavaField\":\"FileType\",\"columnComment\":\"文件类型，如 PDF、EPUB\",\"columnId\":6,\"columnName\":\"file_type\",\"columnType\":\"varchar(50)\",\"createBy\":\"admin\",\"createTime\":\"2025-05-20 16:43:14\",\"dictType\":\"\",\"edit\":true,\"htmlType\":\"select\",\"increment\":false,\"insert\":tru', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 02:52:51', 74);
INSERT INTO `sys_oper_log` VALUES (202, '代码生成', 8, 'com.ruoyi.generator.controller.GenController.batchGenCode()', 'GET', 1, 'admin', '研发部门', '/tool/gen/batchGenCode', '127.0.0.1', '内网IP', '{\"tables\":\"book_resource\"}', NULL, 0, NULL, '2025-05-25 02:52:57', 744);
INSERT INTO `sys_oper_log` VALUES (203, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cmd/info/index\",\"createTime\":\"2025-05-24 18:53:14\",\"icon\":\"tab\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2080,\"menuName\":\"说明书\",\"menuType\":\"C\",\"orderNum\":21,\"params\":{},\"parentId\":0,\"path\":\"info\",\"perms\":\"\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"0\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 02:56:36', 28);
INSERT INTO `sys_oper_log` VALUES (204, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/detail/index\",\"createBy\":\"admin\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"图书详情\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2052,\"path\":\"book-detail\",\"query\":\"id\",\"routeName\":\"\",\"status\":\"0\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 10:38:55', 43);
INSERT INTO `sys_oper_log` VALUES (205, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/book/detail/index\",\"createTime\":\"2025-05-25 10:38:55\",\"icon\":\"tree\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2083,\"menuName\":\"图书详情\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2052,\"path\":\"book-detail\",\"perms\":\"\",\"query\":\"id\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 10:40:17', 17);
INSERT INTO `sys_oper_log` VALUES (206, '菜单管理', 1, 'com.ruoyi.web.controller.system.SysMenuController.add()', 'POST', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/video/detail\",\"createBy\":\"admin\",\"icon\":\"tab\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuName\":\"视频详情\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2052,\"path\":\"video-detail\",\"query\":\"id\",\"status\":\"0\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 10:41:22', 10);
INSERT INTO `sys_oper_log` VALUES (207, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/book/detail\",\"createTime\":\"2025-05-25 10:38:55\",\"icon\":\"tab\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2083,\"menuName\":\"图书详情\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2052,\"path\":\"book-detail\",\"perms\":\"\",\"query\":\"id\",\"routeName\":\"\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 10:41:35', 10);
INSERT INTO `sys_oper_log` VALUES (208, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/book/detail\",\"createTime\":\"2025-05-25 10:38:55\",\"icon\":\"tab\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2083,\"menuName\":\"图书详情\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2052,\"path\":\"book-detail\",\"perms\":\"\",\"query\":\"{\\\"storagePath\\\":\\\"\\\", \\\"name\\\",\\\"\\\",fileType:\\\"\\\",audience:\\\"\\\",coverPath:\\\"\\\"}\",\"routeName\":\"BookDetail\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 13:11:01', 16);
INSERT INTO `sys_oper_log` VALUES (209, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/book/detail\",\"createTime\":\"2025-05-25 10:38:55\",\"icon\":\"tab\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2083,\"menuName\":\"图书详情\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2052,\"path\":\"book-detail\",\"perms\":\"\",\"query\":\"{\\\"storagePath\\\":\\\"\\\", \\\"name\\\",\\\"\\\",fileType:\\\"\\\",audience:\\\"\\\",coverPath:\\\"\\\"}\",\"routeName\":\"BookDetail\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 13:14:59', 11);
INSERT INTO `sys_oper_log` VALUES (210, '用户头像', 2, 'com.ruoyi.web.controller.system.SysProfileController.avatar()', 'POST', 1, 'admin', '研发部门', '/system/user/profile/avatar', '127.0.0.1', '内网IP', '', '{\"msg\":\"操作成功\",\"imgUrl\":\"/profile/avatar/2025/05/25/DC_2012-08-06_28220_20250525142611A001.jpg\",\"code\":200}', 0, NULL, '2025-05-25 14:26:11', 114);
INSERT INTO `sys_oper_log` VALUES (211, '菜单管理', 2, 'com.ruoyi.web.controller.system.SysMenuController.edit()', 'PUT', 1, 'admin', '研发部门', '/system/menu', '127.0.0.1', '内网IP', '{\"children\":[],\"component\":\"cms/book/detail\",\"createTime\":\"2025-05-25 10:38:55\",\"icon\":\"tab\",\"isCache\":\"0\",\"isFrame\":\"1\",\"menuId\":2083,\"menuName\":\"图书详情\",\"menuType\":\"C\",\"orderNum\":2,\"params\":{},\"parentId\":2052,\"path\":\"book-detail\",\"perms\":\"\",\"query\":\"id\",\"routeName\":\"BookDetail\",\"status\":\"0\",\"updateBy\":\"admin\",\"visible\":\"1\"}', '{\"msg\":\"操作成功\",\"code\":200}', 0, NULL, '2025-05-25 16:55:00', 19);

-- ----------------------------
-- Table structure for sys_post
-- ----------------------------
DROP TABLE IF EXISTS `sys_post`;
CREATE TABLE `sys_post`  (
  `post_id` bigint NOT NULL AUTO_INCREMENT COMMENT '岗位ID',
  `post_code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '岗位编码',
  `post_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '岗位名称',
  `post_sort` int NOT NULL COMMENT '显示顺序',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '状态（0正常 1停用）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`post_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 5 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '岗位信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_post
-- ----------------------------
INSERT INTO `sys_post` VALUES (1, 'ceo', '董事长', 1, '0', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_post` VALUES (2, 'se', '项目经理', 2, '0', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_post` VALUES (3, 'hr', '人力资源', 3, '0', 'admin', '2025-05-17 15:01:03', '', NULL, '');
INSERT INTO `sys_post` VALUES (4, 'user', '普通员工', 4, '0', 'admin', '2025-05-17 15:01:03', '', NULL, '');

-- ----------------------------
-- Table structure for sys_role
-- ----------------------------
DROP TABLE IF EXISTS `sys_role`;
CREATE TABLE `sys_role`  (
  `role_id` bigint NOT NULL AUTO_INCREMENT COMMENT '角色ID',
  `role_name` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '角色名称',
  `role_key` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '角色权限字符串',
  `role_sort` int NOT NULL COMMENT '显示顺序',
  `data_scope` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '1' COMMENT '数据范围（1：全部数据权限 2：自定数据权限 3：本部门数据权限 4：本部门及以下数据权限）',
  `menu_check_strictly` tinyint(1) NULL DEFAULT 1 COMMENT '菜单树选择项是否关联显示',
  `dept_check_strictly` tinyint(1) NULL DEFAULT 1 COMMENT '部门树选择项是否关联显示',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '角色状态（0正常 1停用）',
  `del_flag` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '删除标志（0代表存在 2代表删除）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`role_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 100 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '角色信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_role
-- ----------------------------
INSERT INTO `sys_role` VALUES (1, '超级管理员', 'admin', 1, '1', 1, 1, '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '超级管理员');
INSERT INTO `sys_role` VALUES (2, '普通角色', 'common', 2, '2', 1, 1, '0', '0', 'admin', '2025-05-17 15:01:03', '', NULL, '普通角色');

-- ----------------------------
-- Table structure for sys_role_dept
-- ----------------------------
DROP TABLE IF EXISTS `sys_role_dept`;
CREATE TABLE `sys_role_dept`  (
  `role_id` bigint NOT NULL COMMENT '角色ID',
  `dept_id` bigint NOT NULL COMMENT '部门ID',
  PRIMARY KEY (`role_id`, `dept_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '角色和部门关联表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_role_dept
-- ----------------------------
INSERT INTO `sys_role_dept` VALUES (2, 100);
INSERT INTO `sys_role_dept` VALUES (2, 101);
INSERT INTO `sys_role_dept` VALUES (2, 105);

-- ----------------------------
-- Table structure for sys_role_menu
-- ----------------------------
DROP TABLE IF EXISTS `sys_role_menu`;
CREATE TABLE `sys_role_menu`  (
  `role_id` bigint NOT NULL COMMENT '角色ID',
  `menu_id` bigint NOT NULL COMMENT '菜单ID',
  PRIMARY KEY (`role_id`, `menu_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '角色和菜单关联表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_role_menu
-- ----------------------------
INSERT INTO `sys_role_menu` VALUES (2, 100);
INSERT INTO `sys_role_menu` VALUES (2, 101);
INSERT INTO `sys_role_menu` VALUES (2, 102);
INSERT INTO `sys_role_menu` VALUES (2, 103);
INSERT INTO `sys_role_menu` VALUES (2, 104);
INSERT INTO `sys_role_menu` VALUES (2, 105);
INSERT INTO `sys_role_menu` VALUES (2, 106);
INSERT INTO `sys_role_menu` VALUES (2, 107);
INSERT INTO `sys_role_menu` VALUES (2, 108);
INSERT INTO `sys_role_menu` VALUES (2, 109);
INSERT INTO `sys_role_menu` VALUES (2, 110);
INSERT INTO `sys_role_menu` VALUES (2, 111);
INSERT INTO `sys_role_menu` VALUES (2, 112);
INSERT INTO `sys_role_menu` VALUES (2, 113);
INSERT INTO `sys_role_menu` VALUES (2, 114);
INSERT INTO `sys_role_menu` VALUES (2, 115);
INSERT INTO `sys_role_menu` VALUES (2, 116);
INSERT INTO `sys_role_menu` VALUES (2, 117);
INSERT INTO `sys_role_menu` VALUES (2, 500);
INSERT INTO `sys_role_menu` VALUES (2, 501);
INSERT INTO `sys_role_menu` VALUES (2, 1000);
INSERT INTO `sys_role_menu` VALUES (2, 1001);
INSERT INTO `sys_role_menu` VALUES (2, 1002);
INSERT INTO `sys_role_menu` VALUES (2, 1003);
INSERT INTO `sys_role_menu` VALUES (2, 1004);
INSERT INTO `sys_role_menu` VALUES (2, 1005);
INSERT INTO `sys_role_menu` VALUES (2, 1006);
INSERT INTO `sys_role_menu` VALUES (2, 1007);
INSERT INTO `sys_role_menu` VALUES (2, 1008);
INSERT INTO `sys_role_menu` VALUES (2, 1009);
INSERT INTO `sys_role_menu` VALUES (2, 1010);
INSERT INTO `sys_role_menu` VALUES (2, 1011);
INSERT INTO `sys_role_menu` VALUES (2, 1012);
INSERT INTO `sys_role_menu` VALUES (2, 1013);
INSERT INTO `sys_role_menu` VALUES (2, 1014);
INSERT INTO `sys_role_menu` VALUES (2, 1015);
INSERT INTO `sys_role_menu` VALUES (2, 1016);
INSERT INTO `sys_role_menu` VALUES (2, 1017);
INSERT INTO `sys_role_menu` VALUES (2, 1018);
INSERT INTO `sys_role_menu` VALUES (2, 1019);
INSERT INTO `sys_role_menu` VALUES (2, 1020);
INSERT INTO `sys_role_menu` VALUES (2, 1021);
INSERT INTO `sys_role_menu` VALUES (2, 1022);
INSERT INTO `sys_role_menu` VALUES (2, 1023);
INSERT INTO `sys_role_menu` VALUES (2, 1024);
INSERT INTO `sys_role_menu` VALUES (2, 1025);
INSERT INTO `sys_role_menu` VALUES (2, 1026);
INSERT INTO `sys_role_menu` VALUES (2, 1027);
INSERT INTO `sys_role_menu` VALUES (2, 1028);
INSERT INTO `sys_role_menu` VALUES (2, 1029);
INSERT INTO `sys_role_menu` VALUES (2, 1030);
INSERT INTO `sys_role_menu` VALUES (2, 1031);
INSERT INTO `sys_role_menu` VALUES (2, 1032);
INSERT INTO `sys_role_menu` VALUES (2, 1033);
INSERT INTO `sys_role_menu` VALUES (2, 1034);
INSERT INTO `sys_role_menu` VALUES (2, 1035);
INSERT INTO `sys_role_menu` VALUES (2, 1036);
INSERT INTO `sys_role_menu` VALUES (2, 1037);
INSERT INTO `sys_role_menu` VALUES (2, 1038);
INSERT INTO `sys_role_menu` VALUES (2, 1039);
INSERT INTO `sys_role_menu` VALUES (2, 1040);
INSERT INTO `sys_role_menu` VALUES (2, 1041);
INSERT INTO `sys_role_menu` VALUES (2, 1042);
INSERT INTO `sys_role_menu` VALUES (2, 1043);
INSERT INTO `sys_role_menu` VALUES (2, 1044);
INSERT INTO `sys_role_menu` VALUES (2, 1045);
INSERT INTO `sys_role_menu` VALUES (2, 1046);
INSERT INTO `sys_role_menu` VALUES (2, 1047);
INSERT INTO `sys_role_menu` VALUES (2, 1048);
INSERT INTO `sys_role_menu` VALUES (2, 1049);
INSERT INTO `sys_role_menu` VALUES (2, 1050);
INSERT INTO `sys_role_menu` VALUES (2, 1051);
INSERT INTO `sys_role_menu` VALUES (2, 1052);
INSERT INTO `sys_role_menu` VALUES (2, 1053);
INSERT INTO `sys_role_menu` VALUES (2, 1054);
INSERT INTO `sys_role_menu` VALUES (2, 1055);
INSERT INTO `sys_role_menu` VALUES (2, 1056);
INSERT INTO `sys_role_menu` VALUES (2, 1057);
INSERT INTO `sys_role_menu` VALUES (2, 1058);
INSERT INTO `sys_role_menu` VALUES (2, 1059);
INSERT INTO `sys_role_menu` VALUES (2, 1060);
INSERT INTO `sys_role_menu` VALUES (2, 2039);
INSERT INTO `sys_role_menu` VALUES (2, 2040);
INSERT INTO `sys_role_menu` VALUES (2, 2041);
INSERT INTO `sys_role_menu` VALUES (2, 2042);
INSERT INTO `sys_role_menu` VALUES (2, 2043);
INSERT INTO `sys_role_menu` VALUES (2, 2044);
INSERT INTO `sys_role_menu` VALUES (2, 2045);
INSERT INTO `sys_role_menu` VALUES (2, 2046);
INSERT INTO `sys_role_menu` VALUES (2, 2047);
INSERT INTO `sys_role_menu` VALUES (2, 2048);
INSERT INTO `sys_role_menu` VALUES (2, 2049);
INSERT INTO `sys_role_menu` VALUES (2, 2050);
INSERT INTO `sys_role_menu` VALUES (2, 2051);
INSERT INTO `sys_role_menu` VALUES (2, 2052);
INSERT INTO `sys_role_menu` VALUES (2, 2067);
INSERT INTO `sys_role_menu` VALUES (2, 2068);
INSERT INTO `sys_role_menu` VALUES (2, 2069);
INSERT INTO `sys_role_menu` VALUES (2, 2070);
INSERT INTO `sys_role_menu` VALUES (2, 2071);
INSERT INTO `sys_role_menu` VALUES (2, 2072);
INSERT INTO `sys_role_menu` VALUES (2, 2073);
INSERT INTO `sys_role_menu` VALUES (2, 2074);
INSERT INTO `sys_role_menu` VALUES (2, 2075);
INSERT INTO `sys_role_menu` VALUES (2, 2076);
INSERT INTO `sys_role_menu` VALUES (2, 2077);
INSERT INTO `sys_role_menu` VALUES (2, 2078);
INSERT INTO `sys_role_menu` VALUES (2, 2080);
INSERT INTO `sys_role_menu` VALUES (2, 2081);
INSERT INTO `sys_role_menu` VALUES (2, 2082);
INSERT INTO `sys_role_menu` VALUES (2, 2083);
INSERT INTO `sys_role_menu` VALUES (2, 2084);

-- ----------------------------
-- Table structure for sys_user
-- ----------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user`  (
  `user_id` bigint NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `dept_id` bigint NULL DEFAULT NULL COMMENT '部门ID',
  `user_name` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '用户账号',
  `nick_name` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NOT NULL COMMENT '用户昵称',
  `user_type` varchar(2) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '00' COMMENT '用户类型（00系统用户）',
  `email` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '用户邮箱',
  `phonenumber` varchar(11) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '手机号码',
  `sex` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '用户性别（0男 1女 2未知）',
  `avatar` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '头像地址',
  `password` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '密码',
  `status` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '账号状态（0正常 1停用）',
  `del_flag` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '0' COMMENT '删除标志（0代表存在 2代表删除）',
  `login_ip` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '最后登录IP',
  `login_date` datetime NULL DEFAULT NULL COMMENT '最后登录时间',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`user_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 100 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '用户信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_user
-- ----------------------------

-- ----------------------------
-- Table structure for sys_user_post
-- ----------------------------
DROP TABLE IF EXISTS `sys_user_post`;
CREATE TABLE `sys_user_post`  (
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `post_id` bigint NOT NULL COMMENT '岗位ID',
  PRIMARY KEY (`user_id`, `post_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '用户与岗位关联表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_user_post
-- ----------------------------
INSERT INTO `sys_user_post` VALUES (1, 1);
INSERT INTO `sys_user_post` VALUES (2, 2);

-- ----------------------------
-- Table structure for sys_user_role
-- ----------------------------
DROP TABLE IF EXISTS `sys_user_role`;
CREATE TABLE `sys_user_role`  (
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `role_id` bigint NOT NULL COMMENT '角色ID',
  PRIMARY KEY (`user_id`, `role_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_bin COMMENT = '用户和角色关联表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_user_role
-- ----------------------------
INSERT INTO `sys_user_role` VALUES (1, 1);
INSERT INTO `sys_user_role` VALUES (2, 2);

-- ----------------------------
-- Table structure for video_category
-- ----------------------------
DROP TABLE IF EXISTS `video_category`;
CREATE TABLE `video_category`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '分类名称，如多尺度分解、稀疏表示、深度学习',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 5 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of video_category
-- ----------------------------
INSERT INTO `video_category` VALUES (1, '多尺度分解');
INSERT INTO `video_category` VALUES (2, '稀疏表示');
INSERT INTO `video_category` VALUES (3, '深度学习');
INSERT INTO `video_category` VALUES (4, '图像融合');

-- ----------------------------
-- Table structure for video_resource
-- ----------------------------
DROP TABLE IF EXISTS `video_resource`;
CREATE TABLE `video_resource`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '视频标题',
  `storage_path` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件存储路径',
  `duration` time NOT NULL COMMENT '视频时长',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '视频简介',
  `audience` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '适用人群',
  `cover_path` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '封面图像路径',
  `category_id` int NOT NULL COMMENT '关联 video_category',
  `created_at` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_video_category`(`category_id` ASC) USING BTREE,
  FULLTEXT INDEX `ft_video_name_desc`(`name`, `description`),
  CONSTRAINT `fk_video_cat` FOREIGN KEY (`category_id`) REFERENCES `video_category` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 147 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of video_resource
-- ----------------------------
INSERT INTO `video_resource` VALUES (1, '1-【研究生基本功】从零搭建Vision transformer，并用于模型训练、推理可视化！', '/videos/1-【研究生基本功】从零搭建Vision transformer，并用于模型训练、推理可视化！-480P 标清-AVC.mp4', '00:12:30', '介绍红外与可见光图像融合中的多尺度分解原理与应用。', '初学者', 'http://localhost:8080/profile/image/1748110282918.png', 1, '2025-05-20 16:35:46');
INSERT INTO `video_resource` VALUES (2, '基于稀疏表示的融合算法实战', '/videos/sparse_rep_tutorial.mp4', '00:25:00', '动手实现稀疏表示法在图像融合中的具体步骤。', '中级工程师', 'http://localhost:8080/profile/image/1748110282921.png', 2, '2025-05-20 16:35:46');
INSERT INTO `video_resource` VALUES (3, '深度学习在图像融合中的最新进展', '/videos/dl_fusion_latest.mp4', '00:40:45', '汇总并演示深度神经网络在图像融合领域的前沿方法。', '高级研究者', 'http://localhost:8080/profile/image/1748110283503.png', 3, '2025-05-20 16:35:46');
INSERT INTO `video_resource` VALUES (4, '多尺度分解法概论', '/videos/multiscale_intro.mp4', '00:12:30', '介绍红外与可见光图像融合中的多尺度分解原理与应用。', '初学者', 'http://localhost:8080/profile/image/1748110283816.png', 1, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (5, '基于稀疏表示的融合算法实战', '/videos/sparse_rep_tutorial.mp4', '00:25:00', '动手实现稀疏表示法在图像融合中的具体步骤。', '中级工程师', 'http://localhost:8080/profile/image/1748110283826.png', 2, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (6, '深度学习在图像融合中的最新进展', '/videos/dl_fusion_latest.mp4', '00:40:45', '汇总并演示深度神经网络在图像融合领域的前沿方法。', '高级研究者', 'http://localhost:8080/profile/image/1748110283827.png', 3, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (7, '卷积神经网络基础', '/videos/cnn_basics.mp4', '00:15:00', '讲解卷积神经网络的基本原理及其在图像处理中的应用。', '初学者', 'http://localhost:8080/profile/image/1748110283835.png', 1, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (8, '图像融合技术的发展趋势', '/videos/fusion_trends.mp4', '00:22:10', '分析图像融合技术的最新研究方向和发展趋势。', '研究者', 'http://localhost:8080/profile/image/1748110283862.png', 2, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (9, '基于GAN的图像融合', '/videos/gan_fusion.mp4', '00:35:00', '使用生成对抗网络进行图像融合的详细过程及代码演示。', '高级研究者', 'http://localhost:8080/profile/image/1748110284315.png', 3, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (10, '多模态图像融合', '/videos/multimodal_fusion.mp4', '00:18:45', '介绍如何将红外图像与可见光图像进行多模态融合。', '研究者', 'http://localhost:8080/profile/image/1748110284337.png', 2, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (11, '稀疏编码与图像融合', '/videos/sparse_coding.mp4', '00:28:00', '展示稀疏编码在图像融合中的应用实例。', '中级工程师', 'http://localhost:8080/profile/image/1748110284360.png', 3, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (12, '基于小波变换的图像融合', '/videos/wavelet_fusion.mp4', '00:20:30', '讲解如何利用小波变换进行图像融合。', '初学者', 'http://localhost:8080/profile/image/1748110284726.png', 1, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (13, '医学图像融合技术', '/videos/medical_fusion.mp4', '00:32:15', '医学领域中图像融合技术的应用和优势。', '医生', 'http://localhost:8080/profile/image/1748110284818.png', 4, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (14, '图像融合的性能评估方法', '/videos/performance_eval.mp4', '00:25:40', '介绍常用的图像融合性能评估指标及其计算方法。', '研究者', 'http://localhost:8080/profile/image/1748110285037.png', 2, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (15, '基于深度学习的多任务融合', '/videos/multitask_fusion.mp4', '00:30:55', '展示深度学习在多任务图像融合中的应用。', '高级研究者', 'http://localhost:8080/profile/image/1748110285168.png', 3, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (16, '图像融合的数学基础', '/videos/math_basics.mp4', '00:18:00', '简要介绍图像融合的数学基础理论。', '初学者', NULL, 1, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (17, '遥感图像融合技术', '/videos/remote_fusion.mp4', '00:27:50', '分析遥感图像融合的核心技术和实际案例。', '工程师', NULL, 4, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (18, '基于梯度的图像融合方法', '/videos/gradient_fusion.mp4', '00:19:45', '介绍一种基于梯度的简单图像融合方法。', '初学者', NULL, 1, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (19, '图像融合的实际案例分析', '/videos/case_analysis.mp4', '00:35:20', '通过实际案例分析图像融合技术的应用。', '工程师', NULL, 2, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (143, '基于PCA的图像融合', '/videos/pca_fusion.mp4', '00:24:35', '详细讲解主成分分析在图像融合中的应用。', '中级工程师', NULL, 3, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (144, '基于边缘检测的融合方法', '/videos/edge_fusion.mp4', '00:21:00', '展示如何结合边缘检测进行图像融合。', '初学者', NULL, 1, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (145, '图像融合的可视化技术', '/videos/visualization.mp4', '00:29:10', '讲解如何高效地可视化图像融合结果。', '研究者', NULL, 2, '2025-05-24 21:13:23');
INSERT INTO `video_resource` VALUES (146, '基于变分模型的图像融合', '/videos/variational_fusion.mp4', '00:33:30', '深入探讨变分模型在图像融合中的应用。', '高级研究者', NULL, 3, '2025-05-24 21:13:23');

SET FOREIGN_KEY_CHECKS = 1;
