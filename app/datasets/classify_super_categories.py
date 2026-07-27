"""Gom nhãn hàng TMĐT (Trung/Anh) vào 100 siêu danh mục tiếng Việt."""
from __future__ import annotations

import csv
import argparse
import re
from collections import defaultdict
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = DATASETS_DIR / "category_counts.csv"
DEFAULT_OUT = DATASETS_DIR / "m5product_label_taxonomy.csv"
DEFAULT_SUMMARY = DATASETS_DIR / "m5product_super_category_summary.csv"

# Thứ tự là quan trọng: các quy tắc cụ thể đứng trước quy tắc rộng hơn.
RULES: list[tuple[str, str]] = [
    ("Chăm sóc da mặt", r"资生堂|百雀羚|兰蔻|梵贞|一枝春|黛珂|娇韵诗|SK-II|倩碧|肌肤之钥|莱蔻|自然堂|爱和纯|珂润|悦诗风吟|玉兰油|芙丽芳丝|玖美堂|雅芳|佰草集|美肤宝|露兰姬娜|碧欧泉|水密码|温碧泉|相宜本草|佰珍堂|大宝|欧缇丽|李医生|德妃|洁面|面霜|乳液|精华|爽肤水|化妆水|面膜|眼霜|防晒|隔离霜|去角质|男士洁面|祛痘|护肤|润肤|护肤套装|CEMOY|eaoron|Rellet|FANJEIS"),
    ("Trang điểm", r"口红|唇膏|唇彩|粉底|粉饼|腮红|胭脂|眼影|眼线|睫毛|眉笔|BB霜|CC霜|彩妆|遮瑕|定妆|美妆蛋|化妆刷|指甲油"),
    ("Nước hoa & hương thơm", r"香水|香薰|香料|精油|香氛|熏香|扩香|香薰蜡烛"),
    ("Chăm sóc tóc", r"洗发|洗头|护发|染发|发膜|发油|烫发|梳子|假发|吹风机"),
    ("Chăm sóc cơ thể", r"沐浴|身体乳|止汗|脱毛|洗手液|香波浴液|浴盐|身体护理"),
    ("Chăm sóc răng miệng", r"牙膏|牙刷|口腔|电动牙刷|冲牙器|漱口水"),
    ("Mỹ phẩm dụng cụ & phụ kiện", r"美容仪|化妆镜|化妆包|化妆品收纳|美甲工具|美容工具"),
    ("Sức khỏe & thiết bị y tế", r"血压|血糖|体温计|制氧|呼吸机|轮椅|护理床|医疗|医用|理疗|按摩器|保暖贴|艾灸|拔罐|药品|保健品"),
    ("Kính mắt & kính áp tròng", r"眼镜|墨镜|太阳镜|镜框|隐形眼镜|美瞳"),
    ("Đồ lót & đồ ngủ", r"文胸|内裤|睡衣|睡袍|保暖内衣|家居服|丝袜|打底裤"),
    ("Áo nữ", r"女装|女士上衣|女式.*(衫|衣)|连衣裙|半身裙|旗袍|汉服|礼服|晚装|职业女.*套装|唐装"),
    ("Áo nam", r"男装|男士.*(衫|衣)|男式.*(衫|衣)|T恤|衬衫|POLO衫|背心"),
    ("Quần & váy", r"牛仔裤|休闲裤|西裤|短裤|裤子|裤装|裙裤|保暖裤|塑身美体裤"),
    ("Áo khoác & trang phục giữ ấm", r"羽绒服|皮衣|皮草|风衣|棉衣|棉服|夹克|大衣|外套|保暖上装"),
    ("Đồng phục & trang phục biểu diễn", r"校服|制服|工作服|演出服|舞蹈.*套装|肚皮舞|拉丁舞|酒店工作制服|戏服"),
    ("Giày nữ", r"女鞋|时装鞋|单鞋|高跟鞋|乐福鞋|豆豆鞋|凉鞋|人字拖|马丁靴"),
    ("Giày nam", r"男鞋|正装皮鞋|皮鞋|商务鞋|男.*靴"),
    ("Giày thể thao", r"运动鞋|跑步鞋|篮球鞋|足球鞋|板鞋|休闲鞋|登山鞋|滑板鞋"),
    ("Phụ kiện giày", r"鞋垫|鞋油|鞋刷|鞋带|鞋柜"),
    ("Mũ, khăn & găng tay", r"帽子|围巾|手套|耳罩|口罩|腰带|皮带"),
    ("Túi xách & vali", r"背包|书包|女士包|男包|手提包|钱包|行李箱|旅行箱|拉杆箱|卡包"),
    ("Trang sức", r"手饰|首饰|耳饰|耳环|项链|项坠|吊坠|戒指|手镯|手链|珠宝|奇石|玉石"),
    ("Đồng hồ", r"腕表|手表|钟表|挂钟|闹钟"),
    ("Nội thất phòng khách", r"沙发|茶几|电视柜|酒柜|边柜|斗柜|玄关台|展示柜"),
    ("Nội thất phòng ngủ", r"床$|床架|皮艺床|布艺床|实木床|铁艺.*床|乳胶枕|枕头|床垫|床头柜"),
    ("Nội thất phòng ăn", r"餐桌|餐椅|餐桌椅|吧台|餐边柜"),
    ("Nội thất văn phòng & học tập", r"书桌|办公桌|电脑桌|书柜|文件柜|办公椅|学习桌|书架"),
    ("Tủ & kệ lưu trữ", r"衣柜|储物柜|收纳柜|置物架|货架|鞋柜|门厅.*柜|玄关柜"),
    ("Đồ trang trí nhà cửa", r"油画|石雕|砖雕|装饰画|摆件|壁饰|永生花|仿真花|绢花|花瓶|相框"),
    ("Đèn & chiếu sáng", r"灯光|灯具|筒灯|吊灯|台灯|吸顶灯|射灯|灯带|照明"),
    ("Rèm, thảm & vải gia dụng", r"窗帘|地毯|桌布|沙发套|桌椅套|布艺|靠垫|抱枕"),
    ("Chăn ga gối đệm", r"被子|被套|床单|四件套|床品|睡袋|防踢被|蚊帐"),
    ("Thiết bị phòng tắm", r"马桶|花洒|浴室柜|浴缸|水龙头|卫浴|毛巾架"),
    ("Dụng cụ vệ sinh nhà cửa", r"湿巾|消毒液|洗洁精|洁厕|清洁剂|拖把|扫把|垃圾桶|清洁工具"),
    ("Dụng cụ nhà bếp", r"菜刀|刀具|砧板|锅铲|餐具|碗|盘|筷|保鲜盒|厨房.*工具"),
    ("Nồi, chảo & đồ nấu ăn", r"电炖锅|煲汤锅|电炖盅|锅具|炒锅|压力锅|砂锅|燃气灶|烤盘"),
    ("Bình nước & đồ uống", r"保温杯|水杯|塑杯|茶具|咖啡杯|酒具|水壶"),
    ("Thiết bị lọc nước", r"净水器|饮水机|净水机"),
    ("Điện lạnh & điều hòa", r"冰箱|空调|冷柜|风扇|取暖器|加湿器|除湿机"),
    ("Thiết bị gia dụng nhỏ", r"电热水壶|电饭煲|豆浆机|榨汁机|料理机|微波炉|烤箱|吸尘器|空气净化器|蒸汽"),
    ("Thiết bị điện tử gia đình", r"电视|投影仪|机顶盒|家庭影院|音响|音乐盒|收音机"),
    ("Camera & nhiếp ảnh", r"相机|摄像机|影室灯|摄影|镜头|三脚架|补光灯"),
    ("Âm thanh & nhạc cụ", r"音箱|耳机|麦克风|吉他|钢琴|乐器|电子琴"),
    ("Điện thoại & phụ kiện", r"手机|手机壳|保护套|充电器|充电宝|数据线|电话|智能手表"),
    ("Máy tính & phụ kiện", r"鼠标|键盘|电脑|笔记本|显示器|硬盘|路由器|打印机|U盘"),
    ("Trò chơi điện tử", r"游戏手柄|游戏机|电玩|游戏卡|VR"),
    ("Thiết bị mạng & an ninh", r"门禁机|电子门锁|监控|摄像头|报警器|对讲机"),
    ("Linh kiện & thiết bị điện", r"开关|插座|电线|电缆|断路器|变压器|机械喷嘴|支架"),
    ("Ô tô: phụ tùng & phụ kiện", r"汽车|车载|车蜡|车挂|尾翼|氛围灯|轮胎|机油|坐垫|摩托车头盔"),
    ("Xe đạp & xe điện", r"自行车|电动车|滑板车|平衡车"),
    ("Dụng cụ cầm tay", r"工具箱|扳手|螺丝刀|电钻|锯|钳子|量具"),
    ("Máy móc công nghiệp", r"液压|装卸车|工业|机械设备|发电机|空压机|焊机|机床"),
    ("Vật liệu xây dựng", r"瓷砖|地板|涂料|水泥|门窗|锁具|管材|五金"),
    ("Đồ dùng cho bé", r"纸尿裤|拉拉裤|婴儿|宝宝|奶瓶|学步车|婴儿床|安全座椅"),
    ("Sữa & thực phẩm cho bé", r"奶粉|米粉|宝宝零食|婴幼儿食品"),
    ("Quần áo trẻ em", r"儿童.*(衣|服|裤|裙)|童装|连体衣|童鞋|校服"),
    ("Đồ chơi giáo dục", r"早教|闪卡|潜能开发|科学实验|教学设备|感统训练|益智"),
    ("Đồ chơi mô hình & điều khiển", r"机器人|变形玩具|遥控飞机|遥控车|模型|积木|玩具"),
    ("Búp bê & đồ chơi nhập vai", r"娃娃|玩偶|过家家|毛绒"),
    ("Đồ chơi ngoài trời", r"滑梯|秋千|蹦床|沙滩玩具|风筝"),
    ("Thể thao bóng", r"足球|篮球|羽毛球|乒乓球|网球|高尔夫"),
    ("Tập luyện & thể hình", r"健身|哑铃|瑜伽|跑步机|运动器材|跳绳"),
    ("Cắm trại & dã ngoại", r"露营|旅游|登山帐篷|帐篷|睡垫|野餐|户外"),
    ("Đồ bơi & thể thao nước", r"潜水服|泳衣|泳镜|游泳|冲浪"),
    ("Câu cá & săn bắn", r"钓鱼|鱼竿|鱼饵|渔具|弓箭"),
    ("Thú cưng: thức ăn", r"猫咪.*零食|猫粮|狗粮|宠物食品"),
    ("Thú cưng: chăm sóc", r"猫.*药品|狗.*药品|宠物.*(药|用品|窝|玩具|美容)"),
    ("Gạo, mì & ngũ cốc", r"大米|面条|面粉|米粉|麦片|谷物"),
    ("Thịt, cá & hải sản", r"鱼干|鱼$|鸭肉|牛肉|猪肉|海鲜|腊肉"),
    ("Đồ ăn vặt", r"零食|糖果|果冻|布丁|海苔|糕点|枣类|坚果|饼干|薯片"),
    ("Trà, cà phê & đồ uống", r"铁观音|茶叶|茶$|咖啡|饮料|果汁|奶茶"),
    ("Gia vị & thực phẩm khô", r"调料|酱油|食用油|醋|香料|干货|菌菇"),
    ("Rượu vang & rượu mạnh", r"白酒|葡萄酒|红酒|黄酒|啤酒|酒$"),
    ("Văn phòng phẩm", r"文具|笔|本子|胶水|胶带|文件夹|印章|奖状|证书"),
    ("Sách, tạp chí & âm nhạc", r"书籍|图书|杂志|教材|唱片|CD"),
    ("Đồ thủ công & mỹ thuật", r"手工|绘画|画材|颜料|剪纸|雕刻"),
    ("Thiết bị thương mại", r"商用|收银|展示架|服装展示架|酒架|货柜"),
    ("Đồ cưới & trang trí sự kiện", r"婚庆|婚礼|新娘|婚纱|喜庆|派对"),
    ("Đồ dùng hút thuốc", r"烟斗|烟嘴|烟灰缸|雪茄|烟具"),
    ("Bảo hộ lao động", r"劳保|安全帽|防护服|防护鞋|防护手套|护目镜"),
    ("Phòng cháy chữa cháy", r"灭火器|消防|火警|防毒面具"),
    ("Thiết bị đóng gói & in ấn", r"包装机|封口机|印刷|标签机|打码机"),
    ("Thiết bị nhà hàng & khách sạn", r"酒店用品|餐饮设备|厨房设备|制冰机"),
    ("Thiết bị làm đẹp chuyên nghiệp", r"美容院|美发店|纹身|纹绣|美甲店"),
    ("Linh kiện máy tính", r"显卡|主板|CPU|内存|机箱|散热器"),
    ("Thiết bị văn phòng", r"复印机|碎纸机|考勤机|投影幕|装订机"),
    ("Đồ du lịch & phụ kiện", r"旅行用品|护照夹|旅行收纳|行李牌"),
    ("Nhiên liệu & phụ gia", r"燃油|燃气|润滑油|添加剂|煤气"),
    ("Thẻ, mã nạp & tài khoản số", r"点卡|充值卡|账号|会员卡"),
    ("Thiết bị hàng hải & hàng không", r"船舶|航海|飞机配件|无人机"),
    ("Thiết bị truyền thông", r"广播|通信设备|电话交换|对讲"),
    ("Đồ dùng quân sự & mô phỏng", r"军迷|军品|战术|军服"),
    ("Dịch vụ & vé", r"服务|充值|门票|旅游套餐|培训|租赁"),
    ("Quà tặng & đồ lưu niệm", r"礼品|礼盒|纪念品|工艺品|ZIPPO|打火机"),
    ("Đồ dùng tôn giáo & phong thủy", r"佛|香炉|风水|宗教|法器"),
    ("Thiết bị xét nghiệm & phòng thí nghiệm", r"实验室|实验器材|试剂|显微镜"),
    ("Nông nghiệp & làm vườn", r"种子|花盆|园艺|农资|肥料|灌溉|植物"),
    ("Đồng hồ, thiết bị đo lường", r"仪器|测量|测距|检测|传感器"),
    ("Sưu tầm & đồ cổ", r"收藏|古玩|邮票|钱币|纪念币"),
]

FALLBACK_RULES: list[tuple[str, str]] = [
    ("Chăm sóc da mặt", r"泊泉雅|迪奥|优资莱|雅丽洁|韩美肌|蜕唤美|面部护理|护手霜|眼膜|纯露|花水|卸妆|洗护套装|男士面部|喷雾"),
    ("Trang điểm", r"高光|隔离|妆前|蜜粉|散粉"),
    ("Chăm sóc tóc", r"卷/直发|头发造型|理发器|直发片"),
    ("Chăm sóc cơ thể", r"浴足|私处洗液|香皂"),
    ("Sức khỏe & thiết bị y tế", r"同仁堂|成人用纸尿|护具|清凉油|热水袋"),
    ("Kính mắt & kính áp tròng", r"光学镜|防蓝光|老花镜|放大镜|望远镜"),
    ("Đồ lót & đồ ngủ", r"内衣套装|睡裙|运动袜"),
    ("Áo nữ", r"卫衣|绒衫|抹胸|毛衣|针织衫|上衣|亲子装|时尚套装|民族服装|太极服"),
    ("Áo nam", r"西装$|西服$|西服套装|西服/小西装|马夹|马甲"),
    ("Quần & váy", r"西装裤|正装裤|速干裤"),
    ("Áo khoác & trang phục giữ ấm", r"保暖套装"),
    ("Giày nữ", r"帆布鞋|时装靴|高帮鞋|布鞋|洞洞鞋|雨鞋|皮靴|棉靴|松糕.*鞋|广场舞鞋|现代舞鞋"),
    ("Phụ kiện giày", r"居家棉拖|包头拖|家居拖鞋|凉拖|居家鞋"),
    ("Túi xách & vali", r"男士包袋|腰包|旅行袋|斜挎包|拎包|钥匙包|数码收纳|包带|包挂件"),
    ("Trang sức", r"颈饰|胸针"),
    ("Nội thất phòng khách", r"角几|边几|换鞋凳|摇椅|矮凳|折叠椅|躺椅"),
    ("Nội thất phòng ngủ", r"梳妆台|穿衣镜"),
    ("Nội thất văn phòng & học tập", r"升降台|大班台|主管桌|组合.*工作位|办公屏风|高隔断|隔墙|会议桌|折屏"),
    ("Tủ & kệ lưu trữ", r"衣帽架|多宝格|博古架"),
    ("Đồ trang trí nhà cửa", r"木雕|烛台|灯笼|节日装扮|仿真水果|花艺包装"),
    ("Đèn & chiếu sáng", r"落地灯|壁灯|庭院灯|吊扇|手电筒|头灯|LED灯管"),
    ("Rèm, thảm & vải gia dụng", r"毛巾|浴巾"),
    ("Chăn ga gối đệm", r"蚕丝被|棉花被|化纤被"),
    ("Dụng cụ vệ sinh nhà cửa", r"洗衣液|抽纸|卷筒纸|干洗剂|灭鼠|杀虫剂|管道疏通|家私清洁|蚊香液"),
    ("Dụng cụ nhà bếp", r"剪刀|烧烤炉|烤架|绞肉|碎肉|绞菜|保鲜袋|一次性餐盒|纸杯|吸管杯|水桶"),
    ("Nồi, chảo & đồ nấu ăn", r"电热|火锅|电磁炉|陶炉|养生壶|煎药壶|油烟机|电饼铛"),
    ("Bình nước & đồ uống", r"茶壶|茶杯|公道杯|酒杯|保温壶|酒壶"),
    ("Điện lạnh & điều hòa", r"洗衣机|换气扇|排气扇|新风系统"),
    ("Thiết bị gia dụng nhỏ", r"空气芳香剂"),
    ("Thiết bị điện tử gia đình", r"功放|扩音器"),
    ("Âm thanh & nhạc cụ", r"口琴|效果器"),
    ("Điện thoại & phụ kiện", r"移动电源|智能手环"),
    ("Máy tính & phụ kiện", r"一体机|键鼠套装|USB HUB|闪存卡|DIY兼容机"),
    ("Thiết bị mạng & an ninh", r"报警主机|停车场控制|道闸|报警灯|弱电布线"),
    ("Linh kiện & thiết bị điện", r"电源$|UPS电源|插头|转换插头|接线板|步进电机|万用表"),
    ("Ô tô: phụ tùng & phụ kiện", r"车模|智能车机|行车记录仪|大灯总成|洗车机|车用钥匙|脚踏板|置物袋|补胎|脚垫|防追尾|防撞|洗车水枪|扶手箱|排气管|轮毂|倒车镜|漆面|后备箱垫|摩托车.*灯|车顶"),
    ("Xe đạp & xe điện", r"摩托车整车"),
    ("Dụng cụ cầm tay", r"美工刀|工具车|多功能组合工具"),
    ("Vật liệu xây dựng", r"室内门|进户门|亚克力板|弯头|门腕|艺术玻璃"),
    ("Đồ dùng cho bé", r"奶嘴|安抚奶嘴|四轮推车"),
    ("Đồ chơi giáo dục", r"儿童桌面游戏|练字帖|练字板|书法|宣纸|文房四宝|拼图|拼板|魔方|彩泥|橡皮泥|白模填色|模具彩绘"),
    ("Đồ chơi mô hình & điều khiển", r"遥控动物|遥控.*人物|遥控轨道|陀螺"),
    ("Đồ chơi ngoài trời", r"游艺机|游乐设备|充气用品|水枪|吹泡泡|气球"),
    ("Thể thao bóng", r"飞镖|射击|射箭"),
    ("Tập luyện & thể hình", r"运动护具|滑板|广场舞套装"),
    ("Câu cá & săn bắn", r"鱼线轮|钓竿"),
    ("Thú cưng: thức ăn", r"猫主粮|犬主粮|观赏鱼饲料"),
    ("Thú cưng: chăm sóc", r"笼子|逗猫棒|猫砂|猫抓板|猫砂盆|狗狗|宠物服装"),
    ("Đồ ăn vặt", r"素肉|豆腐干|蛋糕|水果罐头|山楂|花生|瓜子|膨化食品|冰淇淋|薯类|笋类|梅类"),
    ("Trà, cà phê & đồ uống", r"普洱|凤凰单丛|大红袍"),
    ("Gia vị & thực phẩm khô", r"腌制|榨菜|泡菜|酱菜|鸡蛋|苹果"),
    ("Rượu vang & rượu mạnh", r"威士忌|Whiskey|酒盒"),
    ("Văn phòng phẩm", r"请柬|贴纸|宣传单|海报|说明书|名片|线$"),
    ("Đồ thủ công & mỹ thuật", r"缝纫DIY|基础材料"),
    ("Thiết bị thương mại", r"灯箱|广告牌|X展架|易拉宝"),
    ("Nông nghiệp & làm vườn", r"大型绿植|组合盆栽|花架|花几|割草机|草坪机|滴灌"),
    ("Đồng hồ, thiết bị đo lường", r"过滤设备|分离设备|水泵"),
]

RULES.extend(FALLBACK_RULES)
RULES.append(("Khác / chưa xác định", r".*"))

SUPER_CATEGORIES = list(dict.fromkeys(category for category, _ in RULES))
assert len(SUPER_CATEGORIES) == 100, len(SUPER_CATEGORIES)

def classify(label: str) -> str:
    for category, pattern in RULES:
        if re.search(pattern, label, re.IGNORECASE):
            return category
    raise AssertionError("fallback must classify every label")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map M5Product labels to Vietnamese super-categories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE,
                        help="CSV containing at least label and sample_count columns.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help="Output taxonomy CSV.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                        help="Output super-category summary CSV.")
    return parser.parse_args()


args = parse_args()
with args.input.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows or "label" not in rows[0] or "sample_count" not in rows[0]:
    raise SystemExit("Input CSV must contain `label` and `sample_count` columns.")

totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
for row in rows:
    category = classify(row["label"])
    row["sieu_danh_muc_tieng_viet"] = category
    totals[category][0] += 1
    totals[category][1] += int(row["sample_count"])

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=[*rows[0].keys()])
    writer.writeheader()
    writer.writerows(rows)

with args.summary.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["sieu_danh_muc_tieng_viet", "so_nhan", "tong_sample_count"])
    for category in SUPER_CATEGORIES:
        writer.writerow([category, *totals[category]])

print(f"{len(rows)} labels -> {args.output}")
print(f"Super-categories: {len(SUPER_CATEGORIES)}")
print("Unclassified:", totals["Khác / chưa xác định"])
