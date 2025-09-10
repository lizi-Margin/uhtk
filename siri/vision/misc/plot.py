###########################################################################################################
# plot
# def plot_image(image, post_process=True):
#     assert len(image.shape) == 3
#     if post_process:
#         if isinstance(image, torch.Tensor):
#             image.int()
#             image = image.cpu().numpy()
#             # image = image[::-1].transpose((1, 2, 0))
#         elif isinstance(image, np.ndarray):
#             # print(image.shape)
#             image.astype(np.uint8)
#             # image = image[..., ::-1]
#             # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     assert isinstance(image, np.ndarray)
#     cv2.imshow('Image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def plot_yolo_results(results):
#     for index, result in enumerate(results):
#         result_image = result.plot()
#         result_image = result_image[..., ::-1]
#         plot_image(result_image)
#         # cv2.imwrite(f"output_{index}.jpg", result_image)